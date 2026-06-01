import json
import time


CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": (
        "Content-Type, Authorization, authorization, "
        "X-Appwrite-JWT, x-appwrite-jwt"
    ),
}


def get_body(context):
    req = context.req

    body = getattr(req, "body", None)

    if isinstance(body, dict):
        return body

    if isinstance(body, str) and body.strip():
        try:
            return json.loads(body)
        except Exception:
            return {"raw": body}

    return {}


def json_response(context, data, status=200):
    return context.res.json(data, status, CORS_HEADERS)


def text_response(context, text, status=200):
    return context.res.text(text, status, CORS_HEADERS)


def main(context):
    try:
        method = getattr(context.req, "method", "GET")
        path = getattr(context.req, "path", "/") or "/"

        context.log(f"Method: {method}")
        context.log(f"Path: {path}")

        if method == "OPTIONS":
            return text_response(context, "", 200)

        if path == "/" and method in ["GET", "POST"]:
            return json_response(
                context,
                {
                    "success": True,
                    "message": "Hello from HXABYTE AI Engine",
                    "runtime": "python-ml-3.11",
                    "routes": [
                        "/",
                        "/health",
                        "/hello",
                        "/echo",
                        "/ai/music/stem-separate-coming-soon",
                    ],
                    "timestamp": int(time.time()),
                },
            )

        if path == "/health" and method in ["GET", "POST"]:
            return json_response(
                context,
                {
                    "success": True,
                    "status": "ok",
                    "service": "ai-engine",
                    "python_function": "working",
                },
            )

        if path == "/hello" and method in ["GET", "POST"]:
            return text_response(
                context,
                "Hello World from Python Appwrite Function!"
            )

        if path == "/echo" and method == "POST":
            body = get_body(context)

            return json_response(
                context,
                {
                    "success": True,
                    "message": "Echo route working",
                    "received": body,
                },
            )

        return json_response(
            context,
            {
                "success": False,
                "error": "Route not found",
                "path": path,
                "method": method,
            },
            404,
        )

    except Exception as err:
        context.error(str(err))

        return json_response(
            context,
            {
                "success": False,
                "error": "Internal server error",
                "detail": str(err),
            },
            500,
        )