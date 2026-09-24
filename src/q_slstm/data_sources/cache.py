# Raw-response snapshot store.
#
# Every retrieval stores the exact response bytes in a new, never-modified snapshot directory
#     <root>/<key>/<retrieved_at>_<sha256[:12]>/<filename>        (bytes)
#     <root>/<key>/<retrieved_at>_<sha256[:12]>/snapshot.json     (URL, validators, sha256, ...)
# and then atomically repoints <root>/<key>/latest.json at it. Bytes are written before the metadata
# and the pointer is written last, all via write-to-temporary + os.replace, so an interrupted download
# never becomes a cache hit: readers only follow `latest.json` and re-verify the SHA-256 on load.
# `--refresh` therefore creates a new snapshot; datasets built from older snapshots keep their bytes.

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path


class CacheMiss(RuntimeError):
    """No usable snapshot for a key (raised in offline mode instead of downloading)."""


class CacheCorrupt(RuntimeError):
    """A snapshot's bytes no longer match its recorded SHA-256."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


class SnapshotStore:
    def __init__(self, root):
        self.root = Path(root)

    def _key_dir(self, key):
        path = self.root / key
        if ".." in Path(key).parts or Path(key).is_absolute():
            raise ValueError(f"invalid cache key {key!r}")
        return path

    def save(self, key, filename, data, metadata, retrieved_at=None):
        """Store `data` as a new snapshot for `key` and point `latest.json` at it. Returns the record."""
        retrieved_at = retrieved_at or utc_now()
        digest = sha256_bytes(data)
        snap_dir = self._key_dir(key) / f"{retrieved_at.strftime('%Y%m%dT%H%M%SZ')}_{digest[:12]}"
        n = 1
        while snap_dir.exists():  # same second and content (e.g. a fast refresh): keep both
            snap_dir = snap_dir.with_name(f"{snap_dir.name.split('~')[0]}~{n}")
            n += 1
        record = {
            **metadata,
            "key": key,
            "filename": filename,
            "sha256": digest,
            "n_bytes": len(data),
            "retrieved_at_utc": retrieved_at.isoformat().replace("+00:00", "Z"),
            "snapshot": snap_dir.name,
        }
        atomic_write_bytes(snap_dir / filename, data)
        atomic_write_json(snap_dir / "snapshot.json", record)
        atomic_write_json(self._key_dir(key) / "latest.json", {"snapshot": snap_dir.name, "sha256": digest})
        return record

    def latest(self, key):
        """Record of the snapshot `latest.json` points at, or None when there is none."""
        pointer = self._key_dir(key) / "latest.json"
        if not pointer.exists():
            return None
        target = json.loads(pointer.read_text(encoding="utf-8"))
        meta_path = self._key_dir(key) / target["snapshot"] / "snapshot.json"
        if not meta_path.exists():
            raise CacheCorrupt(f"{pointer} points at a missing snapshot {target['snapshot']}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def load(self, record):
        """Bytes of a snapshot record, re-verified against its SHA-256."""
        path = self._key_dir(record["key"]) / record["snapshot"] / record["filename"]
        if not path.exists():
            raise CacheMiss(f"snapshot bytes missing: {path}")
        data = path.read_bytes()
        if sha256_bytes(data) != record["sha256"]:
            raise CacheCorrupt(f"{path} does not match its recorded sha256")
        return data

    def path_of(self, record):
        return self._key_dir(record["key"]) / record["snapshot"] / record["filename"]
