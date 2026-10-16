"""Проверки перед публикацией в GitHub: тесты, статический контроль, секреты и
персональные данные, крупные файлы, состояние рабочего дерева.

    python tools/run_checks.py --title "что проверяем"          # из каталога exchange/
    python tools/run_checks.py --title "..." --no-log           # только вывод

Результат дописывается разделом в docs/VERIFICATION_LOG.md. Код возврата не нулевой,
если упал хотя бы один тест или найден возможный секрет: публиковать в таком состоянии
нельзя.

Интерпретаторы задаются переменными окружения, потому что пакет и исследование живут
в разных окружениях:
    PY_PKG       — окружение пакета exchange с pytest     (по умолчанию: текущий python)
    PY_RESEARCH  — окружение исследования (numba, vectorbt) (по умолчанию: PY_PKG)
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

EXCHANGE = Path(__file__).resolve().parents[1]
LOG = EXCHANGE / "docs" / "VERIFICATION_LOG.md"
RESEARCH_TESTS = ["research/ma_crossover_5m/test_ma.py", "research/ma_crossover_5m/test_kernel.py",
                  "research/ma_crossover_5m/test_robust.py"]

SECRET_PATTERNS = {
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}",
    "ключ API вида sk-": r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}",
    "Slack token": r"xox[baprs]-[A-Za-z0-9-]{10,}",
    "Google API key": r"AIza[0-9A-Za-z_\-]{35}",
    "приватный ключ": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "JWT": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    "пароль/токен в присваивании": (r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|"
                                     r"auth[_-]?token|token)\b\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']"),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_ALLOWED = re.compile(r"(?i)(noreply@anthropic\.com|@example\.(com|org)|@users\.noreply\.github\.com)$")
SENSITIVE_NAMES = re.compile(r"(?i)(^|/)(\.env(\..*)?|id_rsa.*|id_ed25519.*|.*\.pem|.*\.p12|.*\.pfx|"
                             r"\.netrc|credentials.*\.json|.*secret.*\.(json|ya?ml|txt))$")
BINARY_SKIP = {".png", ".jpg", ".jpeg", ".gif", ".parquet", ".npy", ".pkl", ".so", ".whl", ".zip", ".pdf"}
MAX_SCAN = 40_000_000
WARN_SIZE, MAX_SIZE = 50_000_000, 100_000_000


def run(cmd, cwd=EXCHANGE, timeout=3600):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def repo_root() -> Path:
    return Path(run(["git", "rev-parse", "--show-toplevel"]).stdout.strip())


def git_files(root: Path) -> tuple[list[str], list[str]]:
    tracked = run(["git", "ls-files"], cwd=root).stdout.splitlines()
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"], cwd=root).stdout.splitlines()
    return tracked, untracked


def pytest(python: str, args: list[str], cwd: Path = EXCHANGE) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "junit.xml"
        t0 = dt.datetime.now()
        proc = run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *args], cwd=cwd)
        seconds = (dt.datetime.now() - t0).total_seconds()
        if not xml.exists():
            return {"ok": False, "tests": 0, "failures": 0, "errors": 1, "skipped": 0, "seconds": seconds,
                    "note": (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["pytest не запустился"]}
        suite = ET.parse(xml).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        counts = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
        return {"ok": counts["failures"] == 0 and counts["errors"] == 0 and counts["tests"] > 0,
                "seconds": seconds, "note": [], **counts}


def text_of(path: Path) -> str | None:
    """Текст файла для поиска секретов; у xlsx — строки и формулы, у .gz — содержимое."""
    suffix = path.suffix.lower()
    if suffix in BINARY_SKIP or path.stat().st_size > MAX_SCAN:
        return None
    try:
        if suffix in (".xlsx", ".xlsm"):
            with zipfile.ZipFile(path) as z:
                parts = [n for n in z.namelist() if n.endswith(".xml") and
                         (n.startswith("xl/sharedStrings") or n.startswith("docProps") or n == "xl/workbook.xml")]
                return "\n".join(z.read(n).decode("utf-8", "replace") for n in parts)
        if suffix == ".gz":
            with gzip.open(path, "rb") as fh:
                return fh.read(MAX_SCAN).decode("utf-8", "replace")
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            return None
        return raw.decode("utf-8", "replace")
    except (OSError, zipfile.BadZipFile, EOFError):
        return None


def scan(root: Path, files: list[str]) -> dict:
    secrets, emails, names, big, skipped, scanned = [], {}, [], [], 0, 0
    patterns = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > WARN_SIZE:
            big.append((rel, size))
        if SENSITIVE_NAMES.search(rel):
            names.append(rel)
        text = text_of(path)
        if text is None:
            skipped += 1
            continue
        scanned += 1
        for label, rx in patterns.items():
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                secrets.append((rel, line, label, m.group(0)[:12] + "…"))
        for m in EMAIL.finditer(text):
            addr = m.group(0)
            if not EMAIL_ALLOWED.search(addr):
                emails.setdefault(addr, set()).add(rel)
    return {"secrets": secrets, "emails": emails, "names": names, "big": big,
            "scanned": scanned, "skipped": skipped}


def static_check(root: Path) -> dict:
    ruff = shutil.which("ruff")
    if not ruff:
        return {"ok": None, "note": "ruff не найден — пропущено"}
    # Фрагменты ячеек ноутбуков исполняются в общем пространстве имён ноутбука:
    # имена им задаёт первая ячейка, поэтому «неопределённое имя» там — не ошибка.
    proc = run([ruff, "check", "--select", "E9,F63,F7,F82", "--output-format", "concise",
                "--extend-exclude", "**/research/ma_crossover_5m/nb,**/research/seven_families/notebook_cells",
                "exchange"], cwd=root)
    lines = [ln for ln in proc.stdout.splitlines() if ln and not ln.startswith(("Found", "All checks"))]
    return {"ok": proc.returncode == 0, "note": f"{len(lines)} замечаний" if lines else "замечаний нет",
            "lines": lines[:20]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    py_pkg = os.environ.get("PY_PKG", sys.executable)
    py_res = os.environ.get("PY_RESEARCH", py_pkg)
    root = repo_root()
    tracked, untracked = git_files(root)
    status = run(["git", "status", "--short", "--untracked-files=all"], cwd=root).stdout.splitlines()
    head = run(["git", "rev-parse", "--short", "HEAD"], cwd=root).stdout.strip()
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root).stdout.strip()

    print("тесты пакета…", flush=True)
    pkg = pytest(py_pkg, ["tests"])
    print("тесты исследования…", flush=True)
    res = pytest(py_res, RESEARCH_TESTS)
    print("статический контроль…", flush=True)
    static = static_check(root)
    print("секреты и персональные данные…", flush=True)
    found = scan(root, tracked + untracked)

    blocking = (not pkg["ok"] or not res["ok"] or static["ok"] is False or found["secrets"] or found["names"]
                or any(s > MAX_SIZE for _, s in found["big"]))
    mark = lambda ok: "✅" if ok else ("⚠️" if ok is None else "❌")  # noqa: E731

    def test_row(name, r):
        return (f"| {name} | {mark(r['ok'])} | {r['tests']} | {r['failures']} | {r['errors']} | {r['skipped']} "
                f"| {r['seconds']:.0f} с |")

    out = [
        f"## {dt.date.today().isoformat()} · {args.title}",
        "",
        f"Ветка `{branch}`, проверено на коммите `{head}` плюс изменения рабочего дерева: "
        f"изменённых файлов — {len(status) - len(untracked)}, новых — {len(untracked)}.",
        "",
        "| Набор тестов | Итог | Тестов | Падений | Ошибок | Пропущено | Время |",
        "|---|---|---|---|---|---|---|",
        test_row("пакет `exchange` (`tests/`)", pkg),
        test_row("исследование скользящих средних (`test_ma`, `test_kernel`, `test_robust`)", res),
        "",
        f"* Статический контроль (ruff: синтаксис, неопределённые имена; без фрагментов ячеек ноутбуков): "
        f"{mark(static['ok'])} {static['note']}.",
        f"* Поиск секретов: просмотрено {found['scanned']} файлов (xlsx — строки и формулы, .csv.gz — "
        f"содержимое), пропущено двоичных {found['skipped']}; возможных секретов — {len(found['secrets'])}, "
        f"подозрительных имён файлов — {len(found['names'])}.",
        "* Адреса электронной почты в файлах (кроме служебных): "
        + (", ".join(f"`{a}` ({len(f)} файл.)" for a, f in sorted(found["emails"].items())) or "нет") + ".",
        "* Файлы крупнее 50 МБ: " + (", ".join(f"`{p}` ({s / 1e6:.0f} МБ)" for p, s in found["big"]) or "нет")
        + " (предел GitHub — 100 МБ на файл).",
    ]
    for rel, line, label, frag in found["secrets"][:30]:
        out.append(f"  * ❌ `{rel}:{line}` — {label}: `{frag}`")
    for rel in found["names"]:
        out.append(f"  * ❌ подозрительное имя файла: `{rel}`")
    for ln in static.get("lines", []):
        out.append(f"  * {ln}")
    for r, name in ((pkg, "пакет"), (res, "исследование")):
        for ln in r["note"]:
            out.append(f"  * {name}: {ln}")
    out += ["", f"**Итог: {'можно публиковать' if not blocking else 'ПУБЛИКОВАТЬ НЕЛЬЗЯ'}.**", ""]
    text = "\n".join(out)
    print(text)
    if not args.no_log:
        if not LOG.exists():
            LOG.write_text("# Журнал проверок перед публикацией\n\n"
                           "Каждый раздел дописан `tools/run_checks.py` перед коммитом: тесты, статический "
                           "контроль, поиск секретов и персональных данных, крупные файлы. Новые разделы — "
                           "сверху.\n\n")
        head_text, _, rest = LOG.read_text().partition("\n## ")
        LOG.write_text(head_text.rstrip() + "\n\n" + text + ("\n## " + rest if rest else ""))
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
