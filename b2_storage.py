"""
b2_storage.py
Thin S3-compatible wrapper around Backblaze B2, used to persist pipeline
artifacts (narration audio, scene manifest, generated images, final video)
OUTSIDE Streamlit Cloud's ephemeral /tmp filesystem.

Why this exists: /tmp is wiped on every container restart, sleep/wake
cycle, and redeploy — none of which are under this app's control. B2 is
real, persistent storage, so a project's progress survives all of that.

Every function degrades to a harmless no-op (returning False/[]/None) if
B2 isn't configured — the rest of the pipeline works identically with or
without it, just without cross-restart resume.
"""

import os


def _get_secret(name):
    try:
        import streamlit as st
        return st.secrets[name]
    except Exception:
        return None


def is_configured():
    return all(_get_secret(k) for k in
               ["B2_KEY_ID", "B2_APPLICATION_KEY", "B2_BUCKET_NAME", "B2_ENDPOINT_URL"])


_client_cache = {}


def _get_client():
    if "client" in _client_cache:
        return _client_cache["client"]
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=_get_secret("B2_ENDPOINT_URL"),
        aws_access_key_id=_get_secret("B2_KEY_ID"),
        aws_secret_access_key=_get_secret("B2_APPLICATION_KEY"),
    )
    _client_cache["client"] = client
    return client


def _bucket():
    return _get_secret("B2_BUCKET_NAME")


def upload_file(local_path, remote_key):
    if not is_configured() or not os.path.exists(local_path):
        return False
    try:
        _get_client().upload_file(local_path, _bucket(), remote_key)
        return True
    except Exception:
        return False


def download_file(remote_key, local_path):
    if not is_configured():
        return False
    try:
        parent = os.path.dirname(local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        _get_client().download_file(_bucket(), remote_key, local_path)
        return True
    except Exception:
        return False


def list_keys(prefix):
    if not is_configured():
        return []
    try:
        keys = []
        paginator = _get_client().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
    except Exception:
        return []


def key_exists(key):
    if not is_configured():
        return False
    try:
        _get_client().head_object(Bucket=_bucket(), Key=key)
        return True
    except Exception:
        return False


def delete_prefix(prefix):
    """Deletes every object under prefix — used by the Clear button."""
    if not is_configured():
        return False
    try:
        client = _get_client()
        keys = list_keys(prefix)
        if not keys:
            return True
        for i in range(0, len(keys), 1000):  # S3-style batch delete caps at 1000/call
            batch = keys[i:i + 1000]
            client.delete_objects(
                Bucket=_bucket(),
                Delete={"Objects": [{"Key": k} for k in batch]},
            )
        return True
    except Exception:
        return False
