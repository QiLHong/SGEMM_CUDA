#!/usr/bin/env python3
"""Close, inspect and commit one RTX5090 checkpoint; never pushes or rewrites history."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

BRANCH = "results/rtx5090-gemm-reproduction-20261022"
PREFIXES = ("docs/zh/rtx5090/", "results/rtx5090/", "scripts/rtx5090/", "experiments/rtx5090/")
SECRET_RULES = (
    rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----",
    rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
    rb"https?://[^\s/@]+:[^\s/@]+@",
    rb"\bAKIA[0-9A-Z]{16}\b",
    rb"(?im)^\s*(?:token|password|secret|access_key|secret_key)\s*[=:]\s*[\"']?[^\s\"']{12,}",
)
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".py", ".ps1", ".sh", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cmake", ".log", ".ptx", ".sass", ".sha256", ".yml", ".yaml", ".exit", ".inc", ".diff"}


def git(root, *args, binary=False):
    p = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        # Never emit Git's potentially credential-bearing URL diagnostics.
        raise RuntimeError(f"git {args[0]} failed (exit {p.returncode}); inspect locally with credentials redacted")
    return p.stdout if binary else p.stdout.decode("utf-8").strip()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def allowed(rel):
    return rel.startswith(PREFIXES) and ".." not in PurePosixPath(rel).parts


def inspect(root, rel):
    p = root / rel
    if not allowed(rel) or p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
        raise RuntimeError(f"Not an allowed regular task file: {rel}")
    if p.name.startswith(".env") or any(part in {".git", "__pycache__", "node_modules", ".venv"} for part in p.parts):
        raise RuntimeError(f"Excluded file: {rel}")
    if p.stat().st_size > 20 * 1024 * 1024:
        raise RuntimeError(f"Large attachment requires separate upload: {rel}")
    if p.suffix.lower() not in TEXT_SUFFIXES and p.name not in {"SHA256SUMS", "CMakeLists.txt"}:
        raise RuntimeError(f"Review unsupported file type before publication: {rel}")
    before = p.stat()
    data = p.read_bytes()
    after = p.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError(f"File is changing: {rel}")
    if b"\0" in data:
        raise RuntimeError(f"Binary data requires separate upload: {rel}")
    data.decode("utf-8-sig")
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(f"LFS pointer is not an uploaded artifact: {rel}")
    if any(re.search(rule, data) for rule in SECRET_RULES):
        raise RuntimeError(f"Potential credential; review file without printing its contents: {rel}")
    return hashlib.sha256(data).hexdigest(), len(data)


def run(args):
    root = Path(git(Path(args.repo), "rev-parse", "--show-toplevel")).resolve()
    gitdir = Path(git(root, "rev-parse", "--absolute-git-dir"))
    state = gitdir / "rtx5090-publish"
    state.mkdir(exist_ok=True)
    with (state / "checkpoint.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if git(root, "branch", "--show-current") != BRANCH:
            raise RuntimeError("Wrong branch")
        if git(root, "remote", "get-url", "origin") != "https://github.com/QiLHong/SGEMM_CUDA.git":
            raise RuntimeError("Unexpected origin (not printed)")
        if git(root, "diff", "--cached", "--name-only"):
            raise RuntimeError("Existing staged changes; preserve and inspect them first")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", args.checkpoint):
            raise RuntimeError("Checkpoint name must be a plain unique identifier")
        paths = set()
        for item in args.path:
            p = root / item
            if p.is_symlink():
                raise RuntimeError("Symlink input is not accepted")
            candidates = [p] if p.is_file() else sorted(p.rglob("*")) if p.is_dir() else []
            if not candidates:
                raise RuntimeError(f"Missing/empty task path: {item}")
            for candidate in candidates:
                if candidate.is_file() or candidate.is_symlink():
                    paths.add(candidate.relative_to(root).as_posix())
        # A marker is produced only after a runner closes its outputs. PARTIAL is
        # publishable evidence but must never be pooled with complete statistics.
        batches = {"/".join(p.split("/")[:3]) for p in paths if p.startswith("results/rtx5090/")}
        for batch in batches:
            marker_path = root / batch / "BATCH_COMPLETE.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
            if marker.get("closed") is not True or marker.get("status") not in {"COMPLETE", "PARTIAL"}:
                raise RuntimeError(f"Batch has not been closed: {batch}")
            if marker.get("status") == "COMPLETE" and not marker.get("independent_summary_verified", False):
                raise RuntimeError(f"Batch needs independent count/statistics verification: {batch}")
            paths.add(marker_path.relative_to(root).as_posix())
        manifest_rel = f"docs/zh/rtx5090/manifests/{args.checkpoint}.sha256"
        paths.discard(manifest_rel)
        reviewed = {rel: inspect(root, rel) for rel in sorted(paths)}
        manifest = "".join(f"{digest}  {rel}\n" for rel, (digest, _) in reviewed.items())
        target = root / manifest_rel
        if target.exists() and target.read_text(encoding="utf-8") != manifest:
            raise RuntimeError("Checkpoint identifier already has a different manifest; use a new identifier")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(manifest, encoding="utf-8")
        reviewed[manifest_rel] = inspect(root, manifest_rel)
        paths.add(manifest_rel)
        parent = git(root, "rev-parse", "HEAD")
        # -f is deliberately absent: ignored files need explicit human review.
        git(root, "add", "--", *sorted(paths))
        staged_paths = git(root, "diff", "--cached", "--name-only", "-z").split("\0")
        if any(p and p not in paths for p in staged_paths):
            raise RuntimeError("Unexpected staged file")
        for rel, (digest, _) in reviewed.items():
            staged = git(root, "show", ":" + rel, binary=True)
            if hashlib.sha256(staged).hexdigest() != digest:
                raise RuntimeError(f"Staged bytes differ from reviewed bytes: {rel}")
        if git(root, "diff", "--cached", "--name-only"):
            git(root, "commit", "-m", args.message)
        head = git(root, "rev-parse", "HEAD")
        pending = {"status": "PENDING_REMOTE_VERIFICATION", "checkpoint": args.checkpoint,
                   "branch": BRANCH, "head": head, "parent_before_checkpoint": parent,
                   "manifest": manifest_rel, "paths": sorted(paths),
                   "files": {p: {"sha256": d, "size": size} for p, (d, size) in reviewed.items()},
                   "utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        dump(state / (args.checkpoint + ".pending.json"), pending)
        # A bundle is local recovery only, never labelled remote backup.
        bundle = state / (args.checkpoint + ".bundle")
        if parent != head and not bundle.exists():
            git(root, "bundle", "create", str(bundle), f"{parent}..HEAD")
        print(json.dumps({"head": head, "manifest": manifest_rel, "files": len(paths),
                          "pending": str(state / (args.checkpoint + ".pending.json")),
                          "backup_verified": False}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--path", action="append", required=True, help="Explicit task file or directory; repeat as needed")
    try:
        run(parser.parse_args())
    except Exception as exc:
        print(f"CHECKPOINT_BLOCKED: {exc}", file=sys.stderr)
        sys.exit(1)

