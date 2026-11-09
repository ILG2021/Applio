"""HTTP helpers for downloading from user-supplied URLs.

* Only http/https URLs are fetched.
* Hosts that resolve to private, loopback, link-local or otherwise non-public
  addresses are refused (blocks SSRF to the LAN or cloud metadata services such
  as 169.254.169.254). Set APPLIO_ALLOW_PRIVATE_DOWNLOADS=1 to allow them.
* Redirects are followed manually so every hop is checked.
* Server-supplied file names are reduced to a safe base name.
"""

import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlsplit

import requests

MAX_REDIRECTS = 5
DEFAULT_TIMEOUT = (15, 60)  # connect, read
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class UnsafeURLError(ValueError):
    pass


def _private_allowed() -> bool:
    return os.environ.get("APPLIO_ALLOW_PRIVATE_DOWNLOADS") == "1"


def validate_url(url: str) -> str:
    if not isinstance(url, str):
        raise UnsafeURLError("URL must be a string.")
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError(f"Only http(s) URLs are allowed: {url}")
    if not parts.hostname:
        raise UnsafeURLError(f"URL has no host: {url}")
    if _private_allowed():
        return url

    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or None)
    except socket.gaierror as error:
        raise UnsafeURLError(f"Cannot resolve host {parts.hostname}: {error}")
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not address.is_global:
            raise UnsafeURLError(
                f"Refusing to download from non-public address {address} ({parts.hostname})."
            )
    return url


def safe_request(method: str, url: str, **kwargs) -> requests.Response:
    """requests.request with URL validation on the initial URL and every redirect."""
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    kwargs["allow_redirects"] = False
    for _ in range(MAX_REDIRECTS + 1):
        validate_url(url)
        response = requests.request(method, url, **kwargs)
        if response.is_redirect and "location" in response.headers:
            url = urljoin(url, response.headers["location"])
            response.close()
            continue
        return response
    raise UnsafeURLError("Too many redirects.")


def safe_get(url: str, **kwargs) -> requests.Response:
    return safe_request("GET", url, **kwargs)


def safe_head(url: str, **kwargs) -> requests.Response:
    return safe_request("HEAD", url, **kwargs)


def safe_filename(name: str, default: str = "downloaded_file") -> str:
    """Reduce a server- or URL-supplied file name to a plain base name."""
    name = str(name or "").replace("\\", "/").split("/")[-1]
    name = name.split("?", 1)[0].split("#", 1)[0]
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    return name or default
