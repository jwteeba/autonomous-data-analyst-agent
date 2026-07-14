from __future__ import annotations
import os
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def _url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def _error_detail(response) -> str:
    try:
        return response.json().get("detail", response.text)
    except Exception:
        return response.text


def _handle(req_fn, *args, **kwargs):
    try:
        r = req_fn(*args, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Can't reach the backend at {API_BASE_URL}. Is it running?"
    except requests.exceptions.HTTPError as e:
        return None, _error_detail(e.response)
    except requests.exceptions.RequestException as e:
        return None, str(e)


def api_get(path: str, timeout: int = 10):
    return _handle(requests.get, _url(path), timeout=timeout)


def api_post(path: str, json: dict | None = None, files=None, timeout: int = 120):
    return _handle(requests.post, _url(path), json=json, files=files, timeout=timeout)


def api_get_raw(path: str) -> str | None:
    try:
        r = requests.get(_url(path), timeout=15)
        r.raise_for_status()
        return r.text
    except requests.exceptions.RequestException:
        return None


def fetch_image_bytes(url_path: str) -> bytes | None:
    try:
        r = requests.get(_url(url_path), timeout=15)
        r.raise_for_status()
        return r.content
    except requests.exceptions.RequestException:
        return None
