#!/usr/bin/env python3
"""Lightweight repository audit tool for DARA.

This script provides static checks for:
1) Potential secret leaks.
2) Project metadata completeness.
3) Basic Python source inventory.
4) Dataset file overview.

It is intentionally independent from training code paths.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "datasets",
    "outputs",
    "logs",
    "wandb",
}


SECRET_PATTERNS: Dict[str, re.Pattern] = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key_header": re.compile(r"BEGIN (RSA|OPENSSH|EC) PRIVATE KEY"),
    "generic_api_key_assignment": re.compile(r"(?i)(api[_-]?key)\s*[:=]\s*[\"'][^\"']{10,}[\"']"),
    "token_assignment": re.compile(r"(?i)(token|secret|password)\s*[:=]\s*[\"'][^\"']{8,}[\"']"),
    "long_hex_string": re.compile(r"\b[a-fA-F0-9]{32,}\b"),
}


@dataclasses.dataclass
class Finding:
    severity: str
    category: str
    path: str
    line: int
    message: str

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RepoStats:
    python_files: int = 0
    python_lines: int = 0
    markdown_files: int = 0
    markdown_lines: int = 0
    test_files: int = 0
    test_lines: int = 0

    def to_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


def iter_files(root: Path, suffixes: Sequence[str] | None = None) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in DEFAULT_IGNORE_DIRS for part in path.parts):
            continue
        if suffixes and path.suffix not in suffixes:
            continue
        yield path


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def collect_stats(root: Path) -> RepoStats:
    stats = RepoStats()

    for path in iter_files(root):
        if path.suffix == ".py":
            lines = safe_read_text(path).splitlines()
            stats.python_files += 1
            stats.python_lines += len(lines)
            if "tests" in path.parts:
                stats.test_files += 1
                stats.test_lines += len(lines)
        elif path.suffix in {".md", ".markdown"}:
            lines = safe_read_text(path).splitlines()
            stats.markdown_files += 1
            stats.markdown_lines += len(lines)

    return stats


def check_required_files(root: Path) -> List[Finding]:
    required = [
        ("README.md", "high", "Missing README.md"),
        ("LICENSE", "high", "Missing LICENSE"),
        (".gitignore", "medium", "Missing .gitignore"),
        ("requirements.txt", "medium", "Missing requirements.txt"),
    ]
    findings: List[Finding] = []
    for rel, sev, msg in required:
        path = root / rel
        if not path.exists():
            findings.append(Finding(sev, "metadata", rel, 0, msg))
    return findings


def detect_secrets(root: Path, max_file_size_kb: int = 1024) -> List[Finding]:
    findings: List[Finding] = []
    for path in iter_files(root):
        if path.stat().st_size > max_file_size_kb * 1024:
            continue

        text = safe_read_text(path)
        if not text:
            continue

        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for name, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    severity = "high" if name in {"private_key_header", "aws_access_key"} else "medium"
                    findings.append(
                        Finding(
                            severity=severity,
                            category="secrets",
                            path=str(path.relative_to(root)),
                            line=i,
                            message=f"Potential secret pattern detected: {name}",
                        )
                    )
    return findings


def check_env_template(root: Path) -> List[Finding]:
    findings: List[Finding] = []
    env_example = root / ".env.example"
    if not env_example.exists():
        findings.append(Finding("medium", "metadata", ".env.example", 0, "Missing .env.example"))
        return findings

    text = safe_read_text(env_example)
    required_keys = ["DARA_MODEL_PATH", "WANDB_MODE", "WANDB_PROJECT"]
    for key in required_keys:
        if key not in text:
            findings.append(Finding("low", "metadata", ".env.example", 0, f"Missing key template: {key}"))
    return findings


def list_dataset_files(root: Path) -> List[Tuple[str, int]]:
    dataset_dir = root / "datasets"
    if not dataset_dir.exists():
        return []
    files = []
    for path in dataset_dir.rglob("*"):
        if path.is_file():
            files.append((str(path.relative_to(root)), path.stat().st_size))
    files.sort(key=lambda x: x[0])
    return files


def summarize_findings(findings: Sequence[Finding]) -> Dict[str, int]:
    summary = {"high": 0, "medium": 0, "low": 0}
    for item in findings:
        summary[item.severity] = summary.get(item.severity, 0) + 1
    return summary


def render_text_report(
    root: Path,
    stats: RepoStats,
    findings: Sequence[Finding],
    dataset_files: Sequence[Tuple[str, int]],
) -> str:
    lines: List[str] = []
    lines.append("DARA Repository Audit Report")
    lines.append("=" * 30)
    lines.append(f"Root: {root}")
    lines.append("")
    lines.append("Project Stats")
    lines.append("-" * 13)
    lines.append(f"Python files: {stats.python_files}")
    lines.append(f"Python lines: {stats.python_lines}")
    lines.append(f"Markdown files: {stats.markdown_files}")
    lines.append(f"Markdown lines: {stats.markdown_lines}")
    lines.append(f"Test files: {stats.test_files}")
    lines.append(f"Test lines: {stats.test_lines}")
    lines.append("")

    counts = summarize_findings(findings)
    lines.append("Findings Summary")
    lines.append("-" * 16)
    lines.append(f"High: {counts['high']}")
    lines.append(f"Medium: {counts['medium']}")
    lines.append(f"Low: {counts['low']}")
    lines.append("")

    if findings:
        lines.append("Detailed Findings")
        lines.append("-" * 17)
        for f in sorted(findings, key=lambda x: (x.severity, x.path, x.line)):
            location = f"{f.path}:{f.line}" if f.line else f.path
            lines.append(f"[{f.severity.upper()}] [{f.category}] {location} - {f.message}")
        lines.append("")
    else:
        lines.append("Detailed Findings")
        lines.append("-" * 17)
        lines.append("No findings.")
        lines.append("")

    lines.append("Dataset Files")
    lines.append("-" * 13)
    if not dataset_files:
        lines.append("No dataset files detected.")
    else:
        for rel, size in dataset_files:
            lines.append(f"{rel} ({size} bytes)")
    lines.append("")

    lines.append("Suggested Next Actions")
    lines.append("-" * 22)
    lines.append("1. Resolve high-severity findings before publishing.")
    lines.append("2. Remove or rotate any exposed credentials.")
    lines.append("3. Keep .env and local artifacts out of version control.")
    lines.append("4. Add automated test execution to CI if needed.")

    return "\n".join(lines)


def render_json_report(
    root: Path,
    stats: RepoStats,
    findings: Sequence[Finding],
    dataset_files: Sequence[Tuple[str, int]],
) -> str:
    payload = {
        "root": str(root),
        "stats": stats.to_dict(),
        "finding_summary": summarize_findings(findings),
        "findings": [f.to_dict() for f in findings],
        "dataset_files": [{"path": p, "size_bytes": s} for p, s in dataset_files],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_audit(root: Path) -> Tuple[RepoStats, List[Finding], List[Tuple[str, int]]]:
    stats = collect_stats(root)
    findings: List[Finding] = []
    findings.extend(check_required_files(root))
    findings.extend(check_env_template(root))
    findings.extend(detect_secrets(root))
    dataset_files = list_dataset_files(root)
    return stats, findings, dataset_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit repository metadata and secret leaks.")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Repository root directory (default: current directory).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output file path. If omitted, print to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root path does not exist: {root}")
        return 2
    if not root.is_dir():
        print(f"Root path is not a directory: {root}")
        return 2

    stats, findings, dataset_files = run_audit(root)
    if args.format == "json":
        report = render_json_report(root, stats, findings, dataset_files)
    else:
        report = render_text_report(root, stats, findings, dataset_files)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to: {out_path}")
    else:
        print(report)

    summary = summarize_findings(findings)
    if summary["high"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
