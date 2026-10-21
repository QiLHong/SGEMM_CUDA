"""Audit local Markdown links; Git membership is separate from path existence.

External URLs are recorded as not checked. This script does not fetch GitHub,
and it does not claim to verify target content merely because a path exists.
"""
from pathlib import Path
import collections
import hashlib
import json
import re
import subprocess
import unicodedata
import urllib.parse

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
SOURCES = [
    "README.md", "docs/zh/00-阅读路线.md", "results/windows/README.md",
    "docs/zh/26-vitamin-SIMT复现与K30公平对照.md",
    "docs/zh/27-vitamin原入口复跑与寄存器上限实验.md",
]


def slug(text):
    text = re.sub(r"<[^>]+>", "", text.strip().lower())
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return "".join(c for c in text if c in " -_" or unicodedata.category(c)[0] in "LN").replace(" ", "-")


def anchors(path):
    counts = collections.Counter()
    values = set()
    for title in re.findall(r"^#{1,6}\s+(.+)$", path.read_text(encoding="utf-8-sig"), re.M):
        value = slug(title)
        index = counts[value]
        counts[value] += 1
        values.add(value if index == 0 else f"{value}-{index}")
    return values


def main():
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode().split("\0"))
    links, hashes, external = [], [], []
    for rel in SOURCES:
        source = REPO / rel
        raw = source.read_bytes()
        hashes.append(dict(source=rel, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        text = raw.decode("utf-8-sig")
        for match in re.finditer(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)", text):
            label, target = match.groups()
            target = target.strip("<>")
            line = text[:match.start()].count("\n") + 1
            if re.match(r"^[a-zA-Z]+:", target):
                external.append(dict(source=rel, line=line, target=target, status="not fetched"))
                continue
            path_text, _, anchor = target.partition("#")
            path = (source.parent / urllib.parse.unquote(path_text)).resolve() if path_text else source
            resolved_rel = path.relative_to(REPO).as_posix()
            tracked_target = resolved_rel in tracked if path.is_file() else any(
                t.startswith(resolved_rel.rstrip("/") + "/") for t in tracked)
            row = dict(source=rel, line=line, label=label, target=target,
                       resolved=resolved_rel, exists=path.exists(),
                       tracked_target=tracked_target, anchor=anchor, anchor_exists=None)
            if anchor and path.is_file() and path.suffix == ".md":
                row["anchor_exists"] = urllib.parse.unquote(anchor) in anchors(path)
            links.append(row)
    bad = [row for row in links if not row["exists"] or row["anchor_exists"] is False]
    summary = dict(local_links=len(links), missing_paths=sum(not row["exists"] for row in links),
                   anchor_checks=sum(row["anchor_exists"] is not None for row in links),
                   missing_anchors=sum(row["anchor_exists"] is False for row in links),
                   links_to_untracked_targets=sum(not row["tracked_target"] for row in links),
                   external_urls_not_fetched=len(external),
                   scope="Local path existence and Markdown heading anchors, with separate Git membership; no remote fetch",
                   status="PASS" if not bad else "INCOMPLETE")
    for filename, data in (("links-after.json", links), ("source-byte-hashes.json", hashes),
                           ("external-not-checked.json", external), ("summary.json", summary)):
        (OUT / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if bad:
        print(json.dumps(bad, ensure_ascii=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
