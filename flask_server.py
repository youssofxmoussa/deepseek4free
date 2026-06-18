import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dsk.api import DeepSeekAPI, AuthenticationError, RateLimitError, NetworkError, APIError
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app, origins="*")

_api = None

def get_api():
    global _api
    if _api is None:
        token = os.getenv("DEEPSEEK_AUTH_TOKEN")
        if not token:
            raise ValueError("DEEPSEEK_AUTH_TOKEN not set")
        _api = DeepSeekAPI(token)
    return _api

def _strip_search(text: str) -> str:
    return re.sub(r'SEARCHING.*?(?:SEARCHINGFINISHED|FINISHED)', '', text, flags=re.DOTALL).strip()

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.route("/deepseek/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "DeepSeek4Free"})


@app.route("/deepseek/api/chat", methods=["POST"])
def chat():
    data              = request.get_json(force=True) or {}
    message           = (data.get("message") or "").strip()
    thinking          = bool(data.get("thinking", False))
    search            = bool(data.get("search", False))
    stream            = bool(data.get("stream", False))
    session_id        = data.get("session_id") or None
    parent_message_id = data.get("parent_message_id") or None

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        api = get_api()
        if not session_id:
            session_id = api.create_chat_session()

        # ── Streaming mode ──────────────────────────────────────────────
        if stream:
            def generate():
                # Send an immediate ping so the proxy doesn't buffer waiting for data
                yield ": ping\n\n"

                try:
                    for chunk in api.chat_completion(
                        session_id, message,
                        parent_message_id=parent_message_id,
                        thinking_enabled=thinking,
                        search_enabled=search,
                    ):
                        t = chunk.get("type", "")
                        c = chunk.get("content", "")

                        if t == "thinking" and c:
                            yield _sse({"type": "thinking", "content": c})

                        elif t == "text" and c:
                            clean = _strip_search(c)
                            if clean:
                                yield _sse({"type": "text", "content": clean})

                        elif t == "done":
                            mid = chunk.get("message_id")
                            yield _sse({"type": "done", "session_id": session_id, "message_id": mid})

                    yield "data: [DONE]\n\n"

                except AuthenticationError as e:
                    yield _sse({"type": "error", "code": 401, "error": str(e)})
                except RateLimitError:
                    yield _sse({"type": "error", "code": 429, "error": "Rate limit exceeded"})
                except NetworkError as e:
                    yield _sse({"type": "error", "code": 503, "error": str(e)})
                except Exception as e:
                    app.logger.exception("Stream error")
                    yield _sse({"type": "error", "code": 500, "error": str(e)})

            return Response(
                stream_with_context(generate()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control":     "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "Connection":        "keep-alive",
                },
            )

        # ── Non-streaming mode ──────────────────────────────────────────
        text_parts     = []
        thinking_parts = []
        message_id     = None

        for chunk in api.chat_completion(
            session_id, message,
            parent_message_id=parent_message_id,
            thinking_enabled=thinking,
            search_enabled=search,
        ):
            t = chunk.get("type", "")
            c = chunk.get("content", "")
            if t == "text" and c:
                text_parts.append(c)
            elif t == "thinking" and c:
                thinking_parts.append(c)
            elif t == "done":
                message_id = chunk.get("message_id")

        response_text = _strip_search("".join(text_parts))
        thinking_text = "".join(thinking_parts)

        return jsonify({
            "response":     response_text,
            "thinking":     thinking_text,
            "has_thinking": bool(thinking_text),
            "session_id":   session_id,
            "message_id":   message_id,
        })

    except AuthenticationError as e:
        return jsonify({"error": f"Authentication failed: {e}"}), 401
    except RateLimitError:
        return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
    except NetworkError as e:
        return jsonify({"error": f"Network error: {e}"}), 503
    except APIError as e:
        return jsonify({"error": f"API error: {e}"}), 500
    except Exception as e:
        app.logger.exception("Unexpected error")
        return jsonify({"error": f"Internal error: {e}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    # threaded=True is required for SSE streaming to work with Flask's dev server
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
