"""Self-contained S3 I/O for the NLA-final `nla/` prefix on Runpod volume iaxphg9saj.

Vendored (not imported from exp04) so it runs standalone on the GPU pod, which only
gets NLA-final/src/. Mirrors exp04/orchestrator/s3_sync.py's hardened client: long
read_timeout for Runpod's slow/flaky endpoint, adaptive retries, plus a coarse
_retry backoff over transient timeouts.

Runpod S3 constraints honored (BUCKET.md §6): NO delete API (write-once; version key
names to "replace"); raw upload_file ALWAYS overwrites (use it to force-replace a
same-size key, which the size-aware skip would otherwise skip); single PutObject for
objects <= 500 MB (our largest ~94 MB); LIST is slow -> always a narrow Prefix, never
the bucket root or hf-cache.

CLI:
  python -m src.s3_io ls   --prefix nla/
  python -m src.s3_io push --local inputs --key-prefix nla/inputs [--skip-same-size]
  python -m src.s3_io pull --key-prefix nla/fve --dest out/fve
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BUCKET = "iaxphg9saj"
DATACENTER = "US-KS-2"
ENDPOINT = f"https://s3api-{DATACENTER}.runpod.io"
MULTIPART_THRESHOLD = 500 * 1024 * 1024

CONNECT_TIMEOUT = int(os.environ.get("S3_CONNECT_TIMEOUT", "15"))
READ_TIMEOUT = int(os.environ.get("S3_READ_TIMEOUT", "300"))
MAX_ATTEMPTS = int(os.environ.get("S3_MAX_ATTEMPTS", "8"))
_RETRY_CAP_S = 30.0
_RETRYABLE_CLIENT_CODES = frozenset({
    "500", "InternalError", "502", "503", "ServiceUnavailable", "504",
    "SlowDown", "Throttling", "ThrottlingException", "RequestTimeout", "RequestTimeoutException",
})


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE; '#' comments; does not override set vars)."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def make_client():
    load_dotenv()
    access, secret = os.environ.get("S3_ACCESS_KEY"), os.environ.get("S3_SECRET")
    if not access or not secret:
        print("ERROR: S3_ACCESS_KEY / S3_SECRET not in environment (.env not loaded?)",
              file=sys.stderr)
        raise SystemExit(2)
    import boto3
    from botocore.config import Config
    cfg = Config(connect_timeout=CONNECT_TIMEOUT, read_timeout=READ_TIMEOUT,
                 retries={"max_attempts": MAX_ATTEMPTS, "mode": "adaptive"})
    return boto3.client("s3", endpoint_url=ENDPOINT, region_name=DATACENTER,
                        aws_access_key_id=access, aws_secret_access_key=secret, config=cfg)


def _is_retryable(exc) -> bool:
    from botocore.exceptions import (
        ClientError, ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError,
    )
    if isinstance(exc, (ReadTimeoutError, EndpointConnectionError, ConnectTimeoutError)):
        return True
    if isinstance(exc, ClientError):
        code = str(exc.response.get("Error", {}).get("Code", ""))
        http = str(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", ""))
        return code in _RETRYABLE_CLIENT_CODES or http.startswith("5")
    return False


def _retry(fn, *, attempts: int = 5, base: float = 2.0, label: str = ""):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc) or i == attempts - 1:
                raise
            delay = min(base * (2 ** i), _RETRY_CAP_S)
            print(f"[s3] transient {label} ({i + 1}/{attempts}): {type(exc).__name__}: {exc} "
                  f"-- retry in {delay:.0f}s", file=sys.stderr, flush=True)
            time.sleep(delay)


def iter_objects(s3, prefix: str):
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = _retry(lambda: s3.list_objects_v2(**kw), label=f"LIST {prefix}")
        yield from resp.get("Contents", [])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")


def head_size(s3, key: str):
    from botocore.exceptions import ClientError
    try:
        return _retry(lambda: s3.head_object(Bucket=BUCKET, Key=key), label=f"HEAD {key}")["ContentLength"]
    except ClientError:
        return None


def upload(s3, local: str | Path, key: str, *, skip_same_size: bool = False) -> bool:
    """PutObject local -> key. Raw upload_file ALWAYS overwrites (Runpod has no delete).
    skip_same_size=True replicates s3_sync's size-aware skip (idempotent re-push)."""
    from boto3.s3.transfer import TransferConfig
    local = Path(local)
    if skip_same_size and head_size(s3, key) == local.stat().st_size:
        print(f"[s3] skip (same size) {key}", flush=True)
        return False
    cfg = TransferConfig(multipart_threshold=MULTIPART_THRESHOLD, multipart_chunksize=MULTIPART_THRESHOLD)
    _retry(lambda: s3.upload_file(str(local), BUCKET, key, Config=cfg), label=f"PUT {key}")
    print(f"[s3] PUT {key} ({local.stat().st_size} B)", flush=True)
    return True


def download(s3, key: str, local: str | Path) -> None:
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    _retry(lambda: s3.download_file(BUCKET, key, str(local)), label=f"GET {key}")
    print(f"[s3] GET {key} -> {local}", flush=True)


def push_dir(s3, local_dir: str | Path, key_prefix: str, *, skip_same_size: bool = False) -> int:
    """Upload every file under local_dir to {key_prefix}/{relpath} (forward-slash keys)."""
    local_dir = Path(local_dir)
    key_prefix = key_prefix.rstrip("/")
    n = 0
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            if upload(s3, p, f"{key_prefix}/{rel}", skip_same_size=skip_same_size):
                n += 1
    print(f"[s3] push_dir {local_dir} -> {key_prefix}/ : {n} uploaded", flush=True)
    return n


def pull_prefix(s3, key_prefix: str, dest_dir: str | Path) -> int:
    """Download every object under key_prefix into dest_dir, stripping the prefix."""
    key_prefix = key_prefix.rstrip("/")
    dest_dir = Path(dest_dir)
    n = 0
    for obj in iter_objects(s3, key_prefix + "/"):
        key = obj["Key"]
        rel = key[len(key_prefix) + 1:]
        download(s3, key, dest_dir / rel)
        n += 1
    print(f"[s3] pull_prefix {key_prefix}/ -> {dest_dir} : {n} objects", flush=True)
    return n


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="NLA-final S3 I/O (nla/ prefix on iaxphg9saj).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ls = sub.add_parser("ls"); p_ls.add_argument("--prefix", required=True)
    p_push = sub.add_parser("push")
    p_push.add_argument("--local", required=True); p_push.add_argument("--key-prefix", required=True)
    p_push.add_argument("--skip-same-size", action="store_true")
    p_pull = sub.add_parser("pull")
    p_pull.add_argument("--key-prefix", required=True); p_pull.add_argument("--dest", required=True)
    args = ap.parse_args(argv)

    s3 = make_client()
    if args.cmd == "ls":
        if args.prefix.rstrip("/") in ("", "nla", BUCKET) and args.prefix.count("/") < 1:
            pass  # nla/ is a narrow-enough prefix; never list bucket root / hf-cache
        total = 0
        for o in iter_objects(s3, args.prefix):
            print(f"{o['Size']:>12}  {o['Key']}")
            total += 1
        print(f"[s3] {total} objects under {args.prefix}")
        return 0
    if args.cmd == "push":
        push_dir(s3, args.local, args.key_prefix, skip_same_size=args.skip_same_size)
        return 0
    pull_prefix(s3, args.key_prefix, args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
