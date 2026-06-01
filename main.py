import base64
import json
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import librosa
import numpy as np


CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": (
        "Content-Type, Authorization, authorization, "
        "X-Appwrite-JWT, x-appwrite-jwt"
    ),
}


MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "8"))
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
DEMUCS_SEGMENT = os.environ.get("DEMUCS_SEGMENT", "5")
DEMUCS_TIMEOUT_SECONDS = int(os.environ.get("DEMUCS_TIMEOUT_SECONDS", "840"))


def json_response(context, payload, status=200):
    return context.res.json(payload, status, CORS_HEADERS)


def text_response(context, text, status=200):
    return context.res.text(text, status, CORS_HEADERS)


def get_request_body(context):
    req = context.req

    body = getattr(req, "body", None)

    if isinstance(body, dict):
        return body

    if isinstance(body, str) and body.strip():
        try:
            return json.loads(body)
        except Exception:
            return {"raw": body}

    for attr in ["body_json", "bodyJson", "body_text", "bodyText"]:
        value = getattr(req, attr, None)

        if isinstance(value, dict):
            return value

        if isinstance(value, str) and value.strip():
            try:
                return json.loads(value)
            except Exception:
                return {"raw": value}

    return {}


def safe_filename(name):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(ch for ch in name if ch in allowed)
    return cleaned or "input.mp3"


def strip_data_url_prefix(value):
    if isinstance(value, str) and value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]

    return value


def estimate_base64_size_bytes(base64_string):
    return int((len(base64_string) * 3) / 4)


def normalize_tempo(raw_tempo):
    if isinstance(raw_tempo, np.ndarray):
        if raw_tempo.size == 0:
            return None
        raw_tempo = raw_tempo.flatten()[0]

    try:
        return round(float(raw_tempo), 2)
    except Exception:
        return None


def detect_tempo_and_duration(input_path):
    y, sr = librosa.load(str(input_path), mono=True, sr=22050)

    if y is None or len(y) == 0:
        return None, 0

    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    return normalize_tempo(tempo), round(float(duration), 2)


def file_to_base64_audio_url(path):
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")

    suffix = path.suffix.lower()

    if suffix == ".wav":
        mime = "audio/wav"
    elif suffix == ".mp3":
        mime = "audio/mpeg"
    else:
        mime = "application/octet-stream"

    return f"data:{mime};base64,{encoded}"


def find_demucs_output_folder(output_dir, track_stem):
    possible = [
        output_dir / DEMUCS_MODEL / track_stem,
        output_dir / "htdemucs" / track_stem,
    ]

    for folder in possible:
        if folder.exists():
            return folder

    for folder in output_dir.rglob("*"):
        if not folder.is_dir():
            continue

        if (
            (folder / "drums.wav").exists()
            and (folder / "bass.wav").exists()
            and (folder / "vocals.wav").exists()
            and (folder / "other.wav").exists()
        ):
            return folder

    return None


def run_demucs(context, input_path, output_dir):
    command = [
        "python",
        "-m",
        "demucs",
        "-n",
        DEMUCS_MODEL,
        "--segment",
        str(DEMUCS_SEGMENT),
        "--shifts",
        "0",
        "-j",
        "1",
        "--out",
        str(output_dir),
        str(input_path),
    ]

    context.log("Running Demucs command:")
    context.log(" ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=DEMUCS_TIMEOUT_SECONDS,
    )

    if result.stdout:
        context.log(result.stdout[-3000:])

    if result.stderr:
        context.error(result.stderr[-3000:])

    if result.returncode != 0:
        raise RuntimeError(result.stderr or "Demucs separation failed")

    return True


def handle_root(context):
    return json_response(
        context,
        {
            "success": True,
            "service": "HXABYTE AI Engine",
            "runtime": "python-ml-3.11",
            "routes": [
                "/",
                "/health",
                "/hello",
                "/ai/music/stem-separate",
                "/stem-separate",
            ],
        },
    )


def handle_health(context):
    return json_response(
        context,
        {
            "success": True,
            "status": "ok",
            "service": "ai-engine",
            "ffmpeg": shutil.which("ffmpeg") or "not-found",
            "demucs_model": DEMUCS_MODEL,
            "demucs_segment": DEMUCS_SEGMENT,
            "max_file_mb": MAX_FILE_MB,
        },
    )


def handle_stem_separate(context):
    body = get_request_body(context)

    file_name = body.get("fileName")
    audio_base64 = (
        body.get("audioBase64")
        or body.get("mp3Base64")
        or body.get("fileBase64")
    )

    if not file_name:
        return json_response(
            context,
            {
                "success": False,
                "detail": "Missing fileName",
            },
            400,
        )

    if not audio_base64:
        return json_response(
            context,
            {
                "success": False,
                "detail": "Missing audioBase64",
            },
            400,
        )

    audio_base64 = strip_data_url_prefix(audio_base64)
    approx_bytes = estimate_base64_size_bytes(audio_base64)

    if approx_bytes > MAX_FILE_MB * 1024 * 1024:
        return json_response(
            context,
            {
                "success": False,
                "detail": f"File too large. Keep it under {MAX_FILE_MB}MB for now.",
            },
            400,
        )

    safe_name = safe_filename(file_name)

    context.log(f"Stem separation started for: {safe_name}")
    context.log(f"Approx file size: {round(approx_bytes / 1024 / 1024, 2)} MB")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        input_path = temp_path / safe_name
        output_dir = temp_path / "separated"

        try:
            audio_bytes = base64.b64decode(audio_base64)
            input_path.write_bytes(audio_bytes)
        except Exception:
            return json_response(
                context,
                {
                    "success": False,
                    "detail": "Invalid base64 audio.",
                },
                400,
            )

        tempo, duration = detect_tempo_and_duration(input_path)

        context.log(f"Detected tempo: {tempo}")
        context.log(f"Detected duration: {duration}")

        run_demucs(context, input_path, output_dir)

        stem_folder = find_demucs_output_folder(output_dir, input_path.stem)

        if not stem_folder:
            raise RuntimeError("Demucs output folder not found.")

        stem_paths = {
            "drums": stem_folder / "drums.wav",
            "bass": stem_folder / "bass.wav",
            "vocals": stem_folder / "vocals.wav",
            "instrument": stem_folder / "other.wav",
        }

        missing = [
            stem_name
            for stem_name, stem_path in stem_paths.items()
            if not stem_path.exists()
        ]

        if missing:
            raise RuntimeError(f"Missing stems: {', '.join(missing)}")

        stems = {
            "drums": file_to_base64_audio_url(stem_paths["drums"]),
            "bass": file_to_base64_audio_url(stem_paths["bass"]),
            "vocals": file_to_base64_audio_url(stem_paths["vocals"]),
            "instrument": file_to_base64_audio_url(stem_paths["instrument"]),
        }

        return json_response(
            context,
            {
                "success": True,
                "message": "Stem separation complete",
                "fileName": file_name,
                "tempo": tempo,
                "duration": duration,
                "model": DEMUCS_MODEL,
                "stems": stems,
            },
            200,
        )


def main(context):
    try:
        method = getattr(context.req, "method", "GET")
        path = getattr(context.req, "path", "/") or "/"

        context.log(f"Method: {method}")
        context.log(f"Path: {path}")

        if method == "OPTIONS":
            return text_response(context, "", 200)

        if path == "/" and method in ["GET", "POST"]:
            return handle_root(context)

        if path == "/health" and method in ["GET", "POST"]:
            return handle_health(context)

        if path == "/hello" and method in ["GET", "POST"]:
            return text_response(
                context,
                "Hello World from HXABYTE AI Engine"
            )

        if (
            method == "POST"
            and path in [
                "/stem-separate",
                "/stem/separate",
                "/ai/music/stem-separate",
            ]
        ):
            return handle_stem_separate(context)

        return json_response(
            context,
            {
                "success": False,
                "detail": "Route not found",
                "path": path,
                "method": method,
            },
            404,
        )

    except subprocess.TimeoutExpired:
        context.error("Demucs timed out")

        return json_response(
            context,
            {
                "success": False,
                "detail": "Stem separation timed out. Try a shorter audio file.",
            },
            504,
        )

    except Exception as error:
        context.error(str(error))
        context.error(traceback.format_exc())

        return json_response(
            context,
            {
                "success": False,
                "detail": str(error),
            },
            500,
        )