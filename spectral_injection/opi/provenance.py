"""Provenance d'une execution : tout ce qu'il faut pour la reproduire.

Le depot n'enregistrait rien de tout ca. Les onze elements demandes par le
protocole sont reunis ici en un seul appel, serialise en JSON a cote de
chaque sortie.
"""

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional


def _git() -> Dict[str, Optional[str]]:
    def run(*args):
        try:
            return subprocess.check_output(
                args, cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))),
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:                                   # noqa: BLE001
            return None
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("git", "status", "--porcelain")}


def _versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "pandas", "torch", "transformers",
                "tokenizers", "pyarrow", "spectral_trust", "huggingface_hub"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception:                                   # noqa: BLE001
            out[mod] = "absent"
    return out


def file_digest(path: str, limit: int = 1 << 24) -> Optional[str]:
    """SHA-256 des ``limit`` premiers octets, ou None si absent."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(limit))
    return h.hexdigest()


def capture(model_id: str, tokenizer, config: Dict[str, Any],
            dataset_path: Optional[str] = None,
            seed: Optional[int] = None,
            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Le dictionnaire de provenance complet."""
    tmpl = getattr(tokenizer, "chat_template", None)
    rec = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "model": {
            "id": model_id,
            "revision": getattr(getattr(tokenizer, "init_kwargs", {}), "get",
                                lambda *_: None)("revision"),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "chat_template_sha256": (hashlib.sha256(tmpl.encode()).hexdigest()
                                     if tmpl else None),
            "chat_template_len": len(tmpl) if tmpl else 0,
        },
        "dataset": {
            "path": dataset_path,
            "sha256_head": file_digest(dataset_path) if dataset_path else None,
        },
        "seed": seed,
        "config": dict(config),
        "versions": _versions(),
        "git": _git(),
    }
    if extra:
        rec["extra"] = extra
    return rec


def write(record: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=1, default=str)
    return path
