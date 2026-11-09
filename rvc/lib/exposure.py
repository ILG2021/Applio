"""Helpers to tell whether Applio is reachable from other machines, and to keep
user-supplied paths inside an allowed directory."""

import os
import sys

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _arg_value(key, default=None):
    if key in sys.argv:
        index = sys.argv.index(key) + 1
        if index < len(sys.argv):
            return sys.argv[index]
    return default


def is_exposed() -> bool:
    """True when the UI may be used by someone other than the local user:
    a --share link, a non-loopback bind address, the inference deployment
    entry point, or APPLIO_RESTRICT_PATHS=1 (for setups behind a reverse proxy)."""
    if os.environ.get("APPLIO_RESTRICT_PATHS") == "1":
        return True
    if "--share" in sys.argv:
        return True
    if os.environ.get("APPLIO_INFER_DEPLOYMENT") == "1":
        return True
    server_name = _arg_value(
        "--server-name", os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    )
    return server_name not in LOOPBACK_HOSTS


def path_within(path, root) -> bool:
    """True when ``path`` resolves (following symlinks) to ``root`` or below it."""
    if not isinstance(path, str) or not path.strip():
        return False
    candidate = os.path.abspath(os.path.join(os.getcwd(), path.strip().strip('"')))
    # Lexical check first, without touching the filesystem: resolving a UNC path
    # such as \\attacker\share on Windows would open an SMB connection and leak
    # NTLM credentials (same class of bug as Starlette CVE-2026-48818).
    if not _contains(os.path.abspath(root), candidate):
        return False
    # Then follow symlinks, which is now safe because the path is local to root.
    return _contains(os.path.realpath(root), os.path.realpath(candidate))


def _contains(root, target) -> bool:
    try:
        return os.path.commonpath([root, target]) == root
    except ValueError:  # different drives on Windows
        return False
