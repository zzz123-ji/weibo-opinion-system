# -*- coding: utf-8 -*-
"""07 参赛文档导出：Markdown → HTML（可选直出 PDF）。

背景
----
提交材料要求 PDF/Word，本机无 pandoc。脚本用 python-markdown 把 docs/*.md
与根目录 README.md 导出为带样式的独立 HTML（图片按相对路径随行）；
加 --pdf 时再用本机 Edge 无头模式打印成 PDF，无需第三方工具。

用法
----
    python src/07_export_docs.py                # 导出 docs/*.html + README.html
    python src/07_export_docs.py --pdf          # 同时生成 PDF
    python src/07_export_docs.py --only 01 02   # 只导出文件名含 01/02 的文档

导出前会扫描 {{占位符}} 与【待填】，有残留时打印警告；最终以
08_submission_check.py 的 FAIL 0 为准。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")

EDGE_CANDIDATES = [
    os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
    os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
]

CSS = """
@page { size: A4; margin: 14mm 13mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
  color: #1f2937; max-width: 960px; margin: 0 auto; padding: 30px 36px;
  line-height: 1.78; font-size: 14px; background: #fff; }
h1 { font-size: 25px; line-height: 1.35; border-bottom: 3px solid #1f5fa8;
  padding-bottom: 10px; color: #102a43; margin: 0 0 18px; }
h2 { font-size: 20px; margin: 32px 0 12px; color: #173f6b;
  border-left: 5px solid #1f5fa8; padding-left: 10px; page-break-after: avoid; }
h3 { font-size: 16px; color: #24507e; margin: 24px 0 8px; page-break-after: avoid; }
h4 { font-size: 14px; color: #2f5d8a; margin: 18px 0 6px; }
p { margin: 8px 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12px;
  table-layout: auto; page-break-inside: auto; }
th, td { border: 1px solid #b9c9db; padding: 5px 7px; text-align: left;
  vertical-align: top; word-break: break-word; }
th { background: #eef3fa; color: #173f6b; font-weight: 600; }
tr:nth-child(even) td { background: #fafcff; }
tr { page-break-inside: avoid; }
code, pre { font-family: Consolas, "Courier New", monospace; font-size: 12.5px; }
code { background: #f2f5f9; padding: 1px 4px; border-radius: 3px; }
pre { background: #f5f7fa; border: 1px solid #dde5ef; padding: 11px 13px;
  overflow-x: auto; border-radius: 6px; white-space: pre-wrap; page-break-inside: avoid; }
blockquote { border-left: 4px solid #ffcc66; background: #fffdf5; margin: 12px 0;
  padding: 8px 14px; color: #6b5a20; }
img { max-width: 100%; height: auto; border: 1px solid #ccd8e6;
  border-radius: 4px; margin: 6px 0; page-break-inside: avoid; }
hr { border: none; border-top: 1px dashed #c9d6e4; margin: 24px 0; }
ul, ol { margin: 6px 0 10px; padding-left: 26px; }
li { margin: 3px 0; }
a { color: #1f5fa8; text-decoration: none; }
@media print {
  body { max-width: none; padding: 0; font-size: 11.5px; }
  h1 { font-size: 21px; }
  h2 { font-size: 17px; }
  h3 { font-size: 14px; }
  table { font-size: 10.5px; }
  pre { font-size: 10.5px; }
  a { color: inherit; }
}
"""


def find_edge():
    for p in EDGE_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return shutil.which("msedge") or shutil.which("microsoft-edge")


def md_to_html(md_path: str) -> tuple[str, str, list[str]]:
    text = Path(md_path).read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        output_format="html5",
    )
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html_body, re.S)
    title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else os.path.basename(md_path)
    placeholders = re.findall(r"\{\{[^}]+\}\}|【待填[^】]*】", text)
    html = (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{CSS}</style></head>"
        f"<body>{html_body}</body></html>"
    )
    return html, title, placeholders


def export_pdf(edge: str, html_path: str, pdf_path: str) -> bool:
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    profile = tempfile.mkdtemp(prefix="edge_pdf_")
    try:
        cmd = [
            edge, "--headless", "--disable-gpu", "--no-first-run",
            "--user-data-dir=" + profile,
            "--no-pdf-header-footer",
            "--print-to-pdf=" + pdf_path,
            "--virtual-time-budget=10000",
            Path(html_path).resolve().as_uri(),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=180)
        ok = os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 20_000
        if not ok:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[-500:]
            print(f"       [pdf 失败] Edge 退出码 {proc.returncode}；{err}")
        return ok
    except Exception as e:
        print(f"       [pdf 异常] {type(e).__name__}: {e}")
        return False
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="参赛文档 Markdown → HTML/PDF")
    ap.add_argument("--pdf", action="store_true", help="同时用 Edge 无头导出 PDF")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只处理文件名含这些子串的文档，如 --only 01 02")
    args = ap.parse_args()

    targets = sorted(glob.glob(os.path.join(DOCS, "*.md")))
    readme = os.path.join(ROOT, "README.md")
    if os.path.exists(readme):
        targets.append(readme)
    if args.only:
        targets = [t for t in targets
                   if any(k in os.path.basename(t) for k in args.only)]
    if not targets:
        raise SystemExit("没有匹配的 Markdown 文件")

    edge = find_edge() if args.pdf else None
    if args.pdf and not edge:
        print("[WARN] 未找到 Edge，将只导出 HTML；可用浏览器手动打印 PDF。")

    warn_total = 0
    pdf_ok = pdf_fail = 0
    for md_path in targets:
        html, title, ph = md_to_html(md_path)
        html_path = os.path.splitext(md_path)[0] + ".html"
        Path(html_path).write_text(html, encoding="utf-8")
        print(f"[html] {os.path.relpath(html_path, ROOT)}  ← {os.path.basename(md_path)}")
        if ph:
            warn_total += len(ph)
            print(f"       [WARN] 仍有 {len(ph)} 个占位符：{ph[0][:50]}…")
        if args.pdf and edge:
            pdf_path = os.path.splitext(html_path)[0] + ".pdf"
            if export_pdf(edge, html_path, pdf_path):
                pdf_ok += 1
                print(f"       [pdf] {os.path.relpath(pdf_path, ROOT)}"
                      f"（{os.path.getsize(pdf_path) / 1024:.0f} KB）")
            else:
                pdf_fail += 1

    print(f"\n完成：HTML {len(targets)} 个；PDF 成功 {pdf_ok} 个，失败 {pdf_fail} 个；"
          f"占位符残留 {warn_total} 个。")
    if args.pdf and pdf_fail:
        print("[WARN] 有 PDF 导出失败：检查 Edge 是否可被无头调用，或手动打开 HTML 打印。")
    if warn_total:
        print("[WARN] 请先补齐占位符，再重新导出 PDF。")


if __name__ == "__main__":
    main()
