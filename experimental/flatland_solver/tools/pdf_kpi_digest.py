from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(f"\n\n===== PAGE {i} =====\n")
        pages.append(page.extract_text() or "")
    return "".join(pages)


def _snippets(text: str, pattern: str, radius: int = 140, limit: int = 6) -> list[str]:
    out = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        s = max(0, m.start() - radius)
        e = min(len(text), m.end() + radius)
        snippet = text[s:e].replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet).strip()
        out.append(snippet)
        if len(out) >= limit:
            break
    return out


def build_report(text: str, pdf_name: str) -> str:
    terms = {
        "done_rate": r"done\s*-?\s*rate",
        "deadlock_rate": r"deadlock\s*-?\s*rate|deadlock",
        "total_reward": r"reward",
        "episode_len": r"episode|schritt|steps",
        "entropy": r"entrop",
        "kl_divergence": r"kl\s*-?\s*diverg|kl\s+early",
        "baseline_dla": r"deadlockavoidancepolicy|\bdla\b|baseline",
        "bc_mappo": r"behavior\s+cloning|\bbc\b|mappo|ppo",
    }

    lines = [
        "# Legacy PDF KPI Digest",
        "",
        f"source: {pdf_name}",
        "",
        "## Extracted KPI Signals",
        "",
    ]

    for key, pattern in terms.items():
        snippets = _snippets(text, pattern)
        lines.append(f"### {key}")
        if snippets:
            for s in snippets:
                lines.append(f"- {s}")
        else:
            lines.append("- no direct hit")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract KPI-related snippets from legacy Flatland PDF")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=Path("kpi_digest.md"))
    parser.add_argument("--out-txt", type=Path, default=Path("kpi_raw.txt"))
    args = parser.parse_args()

    text = _extract_text(args.pdf)
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text(text, encoding="utf-8")

    report = build_report(text, args.pdf.name)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")

    print(f"[pdf-kpi] pages_text={args.out_txt}")
    print(f"[pdf-kpi] report={args.out_md}")


if __name__ == "__main__":
    main()
