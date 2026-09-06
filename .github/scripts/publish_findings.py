#!/usr/bin/env python3
"""Publish scanner output into TWO GitHub issues, and sort them properly.

Why two issues:
  `security-findings` previously held everything, and in practice ~96% of it was
  ruff style nits (UP045 union syntax, I001 import order, F401 unused imports).
  The converge fix-loop dispatches off that issue and takes the top N by
  severity, so it spent its cycles rewriting `Optional[str]` while a genuine
  MEDIUM `torch.load` finding sat beyond the 200-item display cutoff. Security
  and style now go to separate issues with separate labels.

Why the sort matters:
  The previous implementation printed "sorted by severity" in the issue body
  while performing no sort at all - the 200-item window was raw insertion order,
  so whichever scanner's SARIF happened to be globbed first filled it. Real
  findings fell off the end and were invisible to both humans and converge.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

MAX_SHOWN = 200
SECURITY_LABEL = "security-findings"
QUALITY_LABEL = "code-quality"
SECURITY_TITLE = "\U0001f512 Security findings - auto-generated"
QUALITY_TITLE = "\U0001f9f9 Code quality findings - auto-generated"

_LEVEL_RANK = {"error": 0, "warning": 1, "note": 2, "none": 3}


def is_test_path(path: str) -> bool:
    base = os.path.basename(path)
    return "/tests/" in path or base.startswith("test_") or base == "conftest.py"


def sort_key(f: dict) -> tuple:
    """Severity first, then production code ahead of test code, then location.

    Test-code findings are deprioritised rather than dropped: a real issue in a
    test is still worth seeing, just never at the expense of a production one.
    """
    return (
        _LEVEL_RANK.get(str(f.get("level", "warning")).lower(), 1),
        1 if is_test_path(f.get("file", "")) else 0,
        str(f.get("file", "")),
        int(f.get("line", 0) or 0),
    )


def strip_workspace(path: str) -> str:
    """Runner paths are absolute (/home/runner/work/<repo>/<repo>/...) which makes
    the issue body unreadable and non-comparable between runs."""
    ws = os.environ.get("GITHUB_WORKSPACE", "")
    if ws and path.startswith(ws):
        return path[len(ws) :].lstrip("/")
    marker = "/work/"
    if marker in path:
        tail = path.split(marker, 1)[1]
        parts = tail.split("/", 2)
        if len(parts) == 3:
            return parts[2]
    return path


def parse_sarif(filepath: str) -> list[dict]:
    findings: list[dict] = []
    try:
        with open(filepath) as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return findings

    for run in data.get("runs", []):
        tool = run.get("tool", {}).get("driver", {}).get("name", "unknown")
        rules = {r.get("id", ""): r for r in run.get("tool", {}).get("driver", {}).get("rules", [])}
        for result in run.get("results", []):
            # SARIF keeps nosec/nosemgrep-suppressed results with a `suppressions`
            # marker rather than omitting them, so without this every justified
            # suppression is re-reported as an open finding forever.
            if result.get("suppressions"):
                continue
            rid = result.get("ruleId", "unknown")
            desc = rules.get(rid, {}).get("shortDescription", {}).get("text", "")
            for loc in result.get("locations", []):
                phys = loc.get("physicalLocation", {})
                findings.append({
                    "tool": tool,
                    "rule_id": rid,
                    "level": result.get("level", "warning"),
                    "message": result.get("message", {}).get("text", ""),
                    "file": strip_workspace(phys.get("artifactLocation", {}).get("uri", "unknown")),
                    "line": phys.get("region", {}).get("startLine", 0),
                    "description": desc,
                })
    return findings


def parse_ruff(filepath: str) -> list[dict]:
    try:
        with open(filepath) as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    out = []
    for item in data:
        out.append({
            "tool": "ruff",
            "rule_id": item.get("code") or "ruff",
            "level": "note",
            "message": item.get("message", ""),
            "file": strip_workspace(item.get("filename", "unknown")),
            "line": (item.get("location") or {}).get("row", 0),
            "description": item.get("url", ""),
        })
    return out


def parse_coverage(filepath: str = "coverage.xml") -> str:
    """Coverage summary line for the issue body, '' when unavailable.

    The old inline publisher reported this and converge's report consumes it, so
    dropping it on the rewrite would have been a silent regression.
    """
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(filepath).getroot()
        rate = float(root.get("line-rate", 0)) * 100
        covered = int(root.get("lines-covered", 0))
        valid = int(root.get("lines-valid", 0))
        return f"**Coverage:** {rate:.1f}% ({covered}/{valid} lines)"
    except Exception:  # noqa: BLE001 - absent/!parseable coverage must not fail the publish
        return ""


def build_body(findings: list[dict], scan_date: str, kind: str) -> str:
    cov = parse_coverage()
    if not findings:
        body = f"\u2705 No {kind} findings \u2014 last scan: {scan_date}"
        return f"{body}\n\n{cov}" if cov else body

    findings.sort(key=sort_key)
    by_tool: dict[str, int] = {}
    by_level: dict[str, int] = {}
    for f in findings:
        by_tool[f["tool"]] = by_tool.get(f["tool"], 0) + 1
        by_level[f["level"]] = by_level.get(f["level"], 0) + 1

    lines = [f"Last scan: {scan_date}", ""]
    lines.append(f"**Total:** {len(findings)}  ")
    lines.append("**By severity:** " + ", ".join(f"{k}={v}" for k, v in sorted(by_level.items())) + "  ")
    lines.append("**By tool:** " + ", ".join(f"{k}={v}" for k, v in sorted(by_tool.items())))
    if cov:
        lines.append(cov)
    lines.append("")

    shown = findings[:MAX_SHOWN]
    if len(findings) > MAX_SHOWN:
        lines.append(
            f"Showing the {MAX_SHOWN} highest-severity of {len(findings)} findings "
            "(production code before test code)."
        )
        lines.append("")
    lines.append("```json")
    lines.append(json.dumps(shown, indent=2))
    lines.append("```")
    return "\n".join(lines)


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    """Run a gh command and REPORT failure instead of swallowing it.

    Every call here used to pass check=False with the result ignored, so the
    script printed "created issue" and exited 0 whether or not anything was
    created. That is exactly how the code-quality split silently never landed:
    `gh issue create --label code-quality` fails when the label does not exist,
    the scan reported success, and no code-quality issue was ever published in
    any of the six repos (found 2026-07-26).
    """
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"::error::{what} failed (exit {res.returncode}): "
              f"{(res.stderr or res.stdout).strip()[:400]}")
    return res


def ensure_label(label: str, description: str, colour: str) -> bool:
    """Make sure `label` exists before an issue tries to use it.

    `gh issue create --label X` is a hard failure when X is missing - it does not
    create the label on the fly.
    """
    res = subprocess.run(
        ["gh", "label", "list", "--json", "name", "--limit", "200"],
        capture_output=True, text=True,
    )
    try:
        existing = {item["name"] for item in json.loads(res.stdout or "[]")}
    except json.JSONDecodeError:
        existing = set()
    if label in existing:
        return True
    created = _run(
        ["gh", "label", "create", label, "--description", description, "--color", colour],
        f"creating label {label!r}",
    )
    if created.returncode == 0:
        print(f"created missing label {label!r}")
        return True
    return False


def publish(title: str, label: str, body: str) -> bool:
    """Create or update the issue for `label`. Returns True on success."""
    body_file = f"/tmp/findings_{label}.md"
    with open(body_file, "w") as fh:
        fh.write(body)

    if not ensure_label(
        label,
        "Automated scanner findings" if label == "security-findings"
        else "Automated code-quality (lint/style) findings",
        "d93f0b" if label == "security-findings" else "0e8a16",
    ):
        print(f"::error::cannot publish {label} findings - label missing and could not be created")
        return False

    # Match on label, not a free-text search: `gh issue list --search` matched on
    # title words and could pick up an unrelated issue.
    res = subprocess.run(
        ["gh", "issue", "list", "--label", label, "--state", "open", "--json", "number", "--limit", "1"],
        capture_output=True, text=True,
    )
    try:
        issues = json.loads(res.stdout) if res.stdout.strip() else []
    except json.JSONDecodeError:
        issues = []

    if issues:
        num = str(issues[0]["number"])
        done = _run(["gh", "issue", "edit", num, "--body-file", body_file],
                    f"updating issue #{num} ({label})")
        if done.returncode == 0:
            print(f"updated issue #{num} ({label})")
    else:
        done = _run(
            ["gh", "issue", "create", "--title", title, "--label", label, "--body-file", body_file],
            f"creating {label} issue",
        )
        if done.returncode == 0:
            print(f"created issue ({label})")
    return done.returncode == 0


def main() -> int:
    scan_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    security: list[dict] = []
    for path in sorted(glob.glob("*.sarif")):
        security.extend(parse_sarif(path))

    quality = parse_ruff("ruff.json")   # matches the ruff job's upload-artifact name

    print(f"security findings: {len(security)} | quality findings: {len(quality)}")
    ok_sec = publish(SECURITY_TITLE, SECURITY_LABEL, build_body(security, scan_date, "security"))
    ok_qual = publish(QUALITY_TITLE, QUALITY_LABEL, build_body(quality, scan_date, "code quality"))

    # Exit non-zero if either issue could not be published. The advisory scan
    # must never block a merge, but a publisher that quietly does nothing is
    # worse than one that fails: converge dispatches off these issues, so a
    # silent failure starves the fix loop while every job shows green.
    if not (ok_sec and ok_qual):
        print("::error::one or more findings issues could not be published")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
