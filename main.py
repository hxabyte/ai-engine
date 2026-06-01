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


MAX_FILE_MB = int(os.environ.get("MAX_FILE_MB", "15"))
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
DEMUCS_SEGMENT = os.environ.get("DEMUCS_SEGMENT", "7")
RETURN_BASE64_STEMS = os.environ.get("RETURN_BASE64_STEMS", "true").lower() == "true"


def json_response(context, payload, status=200):
    return context.res.json(payload, status, CORS_HEADERS)


def parse_body(context):
    body = getattr(context.req, "body", None)

    if not body:
        return {}

    if isinstance(body, dict):
        return body

    if isinstance(body, str):
        try:
            return json.loads(body)
        except Exception:
            return {}

    return {}


def safe_filename(name):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(ch for ch in name if ch in allowed)
    return cleaned or "input.mp3"


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


def detect_tempo(input_path):
    """
    Uses librosa beat tracking.
    librosa beat_track estimates tempo from onset correlation and beat peaks.
    """
    y, sr = librosa.load(str(input_path), mono=True, sr=22050)

    if y is None or len(y) == 0:
        return None, 0

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    return normalize_tempo(tempo), round(float(duration), 2)


def file_to_base64_data_url(path):
    suffix = path.suffix.lower()

    if suffix == ".wav":
        mime = "audio/wav"
    elif suffix == ".mp3":
        mime = "audio/mpeg"
    else:
        mime = "application/octet-stream"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def find_stem_dir(output_dir, track_name):
    possible_paths = [
        output_dir / DEMUCS_MODEL / track_name,
        output_dir / "htdemucs" / track_name,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # fallback: find folder containing vocals/drums/bass/other
    for path in output_dir.rglob("*"):
        if not path.is_dir():
            continue

        if (
            (path / "vocals.wav").exists()
            and (path / "drums.wav").exists()
            and (path / "bass.wav").exists()
            and (path / "other.wav").exists()
        ):
            return path

    return None


def run_demucs(input_path, output_dir, context):
    """
    Runs Demucs CLI.
    --segment keeps memory lower.
    --shifts 0 keeps it faster and lighter.
    --mp3 is avoided for now because WAV is simpler and reliable.
    """
    cmd = [
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

    context.log("Running command: " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("DEMUCS_TIMEOUT_SECONDS", "840")),
    )

    if result.stdout:
        context.log(result.stdout[-3000:])

    if result.stderr:
        context.error(result.stderr[-3000:])

    if result.returncode != 0:
        raise RuntimeError(result.stderr or "Demucs failed")

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
                "/ai/music/stem-separate",
                "/stem-separate",
            ],
        },
    )


def handle_health(context):
    ffmpeg_path = shutil.which("ffmpeg")
    return json_response(
        context,
        {
            "success": True,
            "message": "AI engine health OK",
            "ffmpeg": ffmpeg_path or "not found",
            "demucs_model": DEMUCS_MODEL,
            "max_file_mb": MAX_FILE_MB,
            "return_base64_stems": RETURN_BASE64_STEMS,
        },
    )


def handle_stem_separate(context):
    body = parse_body(context)

    file_name = body.get("fileName")
    audio_base64 = body.get("audioBase64") or body.get("mp3Base64")

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
                "detail": "Missing audioBase64 or mp3Base64",
            },
            400,
        )

    # Remove data URL prefix if frontend sends data:audio/mp3;base64,...
    if "," in audio_base64 and audio_base64.strip().startswith("data:"):
        audio_base64 = audio_base64.split(",", 1)[1]

    approx_bytes = estimate_base64_size_bytes(audio_base64)
    max_bytes = MAX_FILE_MB * 1024 * 1024

    if approx_bytes > max_bytes:
        return json_response(
            context,
            {
                "success": False,
                "detail": f"File too large. Keep it under {MAX_FILE_MB}MB for now.",
            },
            400,
        )

    safe_name = safe_filename(file_name)

    context.log(f"Stem separation requested: {safe_name}")
    context.log(f"Approx upload size: {round(approx_bytes / 1024 / 1024, 2)}MB")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / safe_name
        output_dir = tmp_dir / "separated"

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

        tempo, duration = detect_tempo(input_path)

        context.log(f"Detected tempo: {tempo}")
        context.log(f"Detected duration: {duration}s")

        run_demucs(input_path, output_dir, context)

        stem_dir = find_stem_dir(output_dir, input_path.stem)

        if not stem_dir:
            raise RuntimeError("Could not find Demucs output stem folder.")

        stem_paths = {
            "drums": stem_dir / "drums.wav",
            "bass": stem_dir / "bass.wav",
            "vocals": stem_dir / "vocals.wav",
            "instrument": stem_dir / "other.wav",
        }

        missing = [name for name, path in stem_paths.items() if not path.exists()]

        if missing:
            raise RuntimeError(f"Missing output stems: {', '.join(missing)}")

        if RETURN_BASE64_STEMS:
            stems = {
                name: file_to_base64_data_url(path)
                for name, path in stem_paths.items()
            }
        else:
            stems = {
                name: {
                    "fileName": path.name,
                    "note": "Stem generated, but RETURN_BASE64_STEMS=false. Add Appwrite Storage upload next.",
                }
                for name, path in stem_paths.items()
            }

        return json_response(
            context,
            {
                "success": True,
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
            return context.res.text("", 200, CORS_HEADERS)

        if path == "/" and method in ["GET", "POST"]:
            return handle_root(context)

        if path == "/health" and method in ["GET", "POST"]:
            return handle_health(context)

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
            },
            404,
        )

    except subprocess.TimeoutExpired:
        context.error("Demucs timeout")
        return json_response(
            context,
            {
                "success": False,
                "detail": "Stem separation timed out. Try a shorter audio file.",
            },
            504,
        )

    except Exception as err:
        context.error(str(err))
        context.error(traceback.format_exc())

        return json_response(
            context,
            {
                "success": False,
                "detail": str(err),
            },
            500,
        )