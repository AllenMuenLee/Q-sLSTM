# Standard-library HTTP client for the data collectors.
#
# - Separate connect / read timeouts.
# - At most `max_attempts` attempts per HTTP request (each byte-range chunk is its own request),
#   exponential backoff with jitter, and `Retry-After` for 429 / 503.
# - Temporary failures (network errors, truncated bodies, 429, 500, 502, 503, 504) are retried;
#   permanent failures (other 4xx, malformed responses) fail immediately.
# - `download` fetches large files as validated byte ranges. The IESO server resets long transfers,
#   and its load-balanced nodes may disagree on ETag, so each chunk must be a 206 whose Content-Range
#   continues the previous chunk and whose total length and Last-Modified match the first chunk.

from __future__ import annotations

import email.utils
import http.client
import random
import re
import time
import urllib.parse
from dataclasses import dataclass, field

TRANSIENT_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "q-slstm-solar-collector/1.0 (research data collection; python stdlib)"


class HttpError(RuntimeError):
    """Permanent request failure: do not retry."""


class TransientHttpError(RuntimeError):
    """Temporary failure; `retry_after` (seconds) is honored when the server supplies it."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    connect_timeout: float = 15.0
    read_timeout: float = 60.0
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    jitter: float = 0.5  # delay *= 1 + jitter * U(0, 1)
    max_retry_after: float = 600.0
    min_request_interval: float = 0.0  # politeness delay between consecutive requests

    def __post_init__(self):
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class HttpResponse:
    url: str
    status: int
    headers: dict
    body: bytes
    n_requests: int = 1
    n_retries: int = 0
    notes: list = field(default_factory=list)


def parse_retry_after(value, now=None):
    """Seconds to wait from a Retry-After header (delta-seconds or HTTP-date); None if unusable."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    now = time.time() if now is None else now
    return max(0.0, when.timestamp() - now)


def stdlib_transport(url, headers, connect_timeout, read_timeout):
    """One GET with http.client: returns (status, lower-cased headers, body). Network errors raise OSError."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise HttpError(f"unsupported URL scheme in {url!r}")
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=connect_timeout)
    try:
        conn.connect()
        conn.sock.settimeout(read_timeout)
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body
    finally:
        conn.close()


_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)$")


class HttpClient:
    """GET with bounded retries. `transport`, `sleep`, and `rng` are injectable for offline tests."""

    def __init__(self, policy=None, transport=None, sleep=time.sleep, rng=None, clock=time.monotonic):
        self.policy = policy or RetryPolicy()
        self.transport = transport or stdlib_transport
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.clock = clock
        self._last_request = None

    # -- single request -----------------------------------------------------------------------

    def _backoff(self, attempt, retry_after):
        p = self.policy
        delay = min(p.backoff_max, p.backoff_base * 2 ** (attempt - 1)) * (1.0 + p.jitter * self.rng.random())
        if retry_after is not None:
            if retry_after > p.max_retry_after:
                raise HttpError(f"server asked to retry after {retry_after:.0f}s (> {p.max_retry_after:.0f}s)")
            delay = max(delay, retry_after)
        return delay

    def _throttle(self):
        interval = self.policy.min_request_interval
        if interval > 0 and self._last_request is not None:
            wait = interval - (self.clock() - self._last_request)
            if wait > 0:
                self.sleep(wait)
        self._last_request = self.clock()

    def _once(self, url, headers, validate):
        self._throttle()
        try:
            status, resp_headers, body = self.transport(
                url, {"User-Agent": USER_AGENT, **headers}, self.policy.connect_timeout, self.policy.read_timeout)
        except (HttpError, TransientHttpError):
            raise
        except (OSError, http.client.HTTPException) as exc:  # resets, timeouts, truncated reads
            raise TransientHttpError(f"{type(exc).__name__}: {exc}") from exc
        if status in TRANSIENT_STATUS:
            raise TransientHttpError(f"HTTP {status}", parse_retry_after(resp_headers.get("retry-after")))
        if status >= 400:
            snippet = body[:300].decode("utf-8", "replace")
            raise HttpError(f"HTTP {status} for {url}: {snippet}")
        length = resp_headers.get("content-length")
        if length is not None and length.isdigit() and int(length) != len(body):
            raise TransientHttpError(f"truncated body: {len(body)} of {length} bytes")
        response = HttpResponse(url, status, resp_headers, body)
        if validate is not None:
            validate(response)
        return response

    def request(self, url, params=None, headers=None, validate=None):
        """GET `url` with at most `max_attempts` attempts; returns HttpResponse."""
        if params:
            url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params, safe=",")
        headers = headers or {}
        errors = []
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                response = self._once(url, headers, validate)
                response.n_requests, response.n_retries = attempt, attempt - 1
                response.notes = errors
                return response
            except TransientHttpError as exc:
                errors.append(f"attempt {attempt}: {exc}")
                if attempt == self.policy.max_attempts:
                    break
                self.sleep(self._backoff(attempt, exc.retry_after))
        raise TransientHttpError(f"giving up on {url} after {self.policy.max_attempts} attempts: {errors}")

    # -- chunked download ---------------------------------------------------------------------

    def download(self, url, chunk_size=1 << 20):
        """Fetch a (possibly large) file as validated byte ranges; falls back to a plain 200 body."""
        def check_first(resp):
            if resp.status == 206:
                self._content_range(resp, 0)

        first = self.request(url, headers={"Range": f"bytes=0-{chunk_size - 1}"}, validate=check_first)
        if first.status == 200:  # server ignored the range: the body is the whole file
            first.notes.append("server returned the full body (no range support)")
            return first
        if first.status != 206:
            raise HttpError(f"unexpected HTTP {first.status} for ranged GET of {url}")
        start, end, total = self._content_range(first, 0)
        validator = first.headers.get("last-modified")
        body = bytearray(first.body)
        n_requests, n_retries, notes = first.n_requests, first.n_retries, list(first.notes)

        while len(body) < total:
            offset = len(body)

            def check(resp, offset=offset):
                if resp.status != 206:
                    raise TransientHttpError(f"expected 206 for range at {offset}, got {resp.status}")
                s, e, t = self._content_range(resp, offset)
                if t != total:
                    raise HttpError(f"{url} changed length during download ({total} -> {t})")
                if validator and resp.headers.get("last-modified") != validator:
                    raise HttpError(f"{url} changed (Last-Modified) during download")

            chunk = self.request(url, headers={"Range": f"bytes={offset}-{min(offset + chunk_size, total) - 1}"},
                                 validate=check)
            body += chunk.body
            n_requests += chunk.n_requests
            n_retries += chunk.n_retries
            notes += chunk.notes
        if len(body) != total:
            raise HttpError(f"assembled {len(body)} bytes, expected {total}")
        headers = {k: v for k, v in first.headers.items() if k not in ("content-range", "content-length")}
        headers["content-length"] = str(total)
        return HttpResponse(url, 200, headers, bytes(body), n_requests, n_retries, notes)

    @staticmethod
    def _content_range(resp, expected_start):
        match = _CONTENT_RANGE.match(resp.headers.get("content-range", ""))
        if not match or match.group(3) == "*":
            raise HttpError(f"missing or malformed Content-Range: {resp.headers.get('content-range')!r}")
        start, end, total = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if start != expected_start or end - start + 1 != len(resp.body):
            raise TransientHttpError(f"range mismatch: asked {expected_start}, got {start}-{end} "
                                     f"with {len(resp.body)} bytes")
        return start, end, total
