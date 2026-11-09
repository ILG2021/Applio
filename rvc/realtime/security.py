"""Access control and path validation for the realtime client API.

The realtime API (mounted at ``/api`` in ``--client`` mode) is reachable by any
web page the user visits, because browsers allow cross-origin WebSocket
connections and "simple" POST requests to localhost.  Every request therefore
has to prove it came from the Applio page itself (a per-process token that is
only embedded in the page served by Applio) and come from an allowed Origin.
Paths received from the client are restricted to Applio's own directories.
"""

import os
import secrets
from urllib.parse import urlsplit

from rvc.lib.exposure import path_within

now_dir = os.getcwd()

# Random per-process token; injected into tabs/realtime/main.js by app.py.
API_TOKEN = os.environ.get("APPLIO_REALTIME_TOKEN") or secrets.token_urlsafe(32)
TOKEN_PLACEHOLDER = "__APPLIO_REALTIME_TOKEN__"
TOKEN_HEADER = "x-applio-token"

MODEL_ROOTS = [os.path.join(now_dir, "logs")]
if os.environ.get("APPLIO_INFER_MODEL_ROOT"):
    MODEL_ROOTS.append(os.environ["APPLIO_INFER_MODEL_ROOT"])
EMBEDDER_CUSTOM_ROOT = os.path.join(
    now_dir, "rvc", "models", "embedders", "embedders_custom"
)
RECORD_ROOT = os.path.join(now_dir, "assets", "audios")

MODEL_EXTENSIONS = (".pth", ".onnx")
INDEX_EXTENSIONS = (".index",)
RECORD_FORMATS = {"WAV", "MP3", "FLAC", "OGG", "M4A"}

# Extra origins (e.g. "https://applio.example.com") allowed when Applio sits
# behind a reverse proxy that rewrites the Host header. Comma separated.
EXTRA_ALLOWED_ORIGINS = {
    o.strip().rstrip("/").lower()
    for o in os.environ.get("APPLIO_REALTIME_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
}


class ValidationError(ValueError):
    pass


def inject_token(js_source: str) -> str:
    return js_source.replace(TOKEN_PLACEHOLDER, API_TOKEN)


def token_valid(token) -> bool:
    return isinstance(token, str) and secrets.compare_digest(token, API_TOKEN)


def origin_allowed(headers) -> bool:
    """The Origin must match the Host the request was sent to (same origin)."""
    origin = headers.get("origin")
    if not origin:
        # Non-browser clients do not send Origin; the token still applies.
        return True
    origin = origin.rstrip("/").lower()
    if origin in EXTRA_ALLOWED_ORIGINS:
        return True
    host = (headers.get("host") or "").lower()
    return bool(host) and urlsplit(origin).netloc == host


def _within(path: str, roots) -> bool:
    # path_within checks lexically before resolving, so UNC paths never hit the network.
    return any(path_within(path, root) for root in roots)


def _resolve(path) -> str:
    if not isinstance(path, str):
        raise ValidationError("Path must be a string.")
    cleaned = path.strip().strip('"').strip()
    return os.path.abspath(os.path.join(now_dir, cleaned))


def check_model_path(path) -> str:
    resolved = _resolve(path)
    if not resolved.lower().endswith(MODEL_EXTENSIONS):
        raise ValidationError("Model must be a .pth or .onnx file.")
    if not _within(resolved, MODEL_ROOTS):
        raise ValidationError("Model must be inside the models folder.")
    return path


def check_index_path(path) -> str:
    if not path:
        return path
    resolved = _resolve(path)
    if not resolved.lower().endswith(INDEX_EXTENSIONS):
        raise ValidationError("Index must be a .index file.")
    if not _within(resolved, MODEL_ROOTS):
        raise ValidationError("Index must be inside the models folder.")
    return path


def check_embedder_custom(path) -> str:
    if not path:
        return path
    if not _within(_resolve(path), [EMBEDDER_CUSTOM_ROOT]):
        raise ValidationError("Custom embedder must be inside embedders_custom.")
    return path


def check_record(path, export_format) -> str:
    if str(export_format).upper() not in RECORD_FORMATS:
        raise ValidationError("Unsupported export format.")
    if not path:
        return path
    if not _within(_resolve(path), [RECORD_ROOT]):
        raise ValidationError("Recording path must be inside assets/audios.")
    return path


_PATH_CHECKS = {
    "model_path": check_model_path,
    "index_path": check_index_path,
    "embedder_model_custom": check_embedder_custom,
}


def check_param(key, value):
    """Validate a single realtime parameter coming from the client."""
    check = _PATH_CHECKS.get(key)
    if check is not None:
        check(value)


def check_params(params: dict):
    if not isinstance(params, dict):
        raise ValidationError("Parameters must be an object.")
    for key, value in params.items():
        check_param(key, value)
