from pathlib import Path

MODULE_ORDER = ["mat", "face", "recon", "colmap", "2dgs", "uv", "tex"]


def parse_progress(log_path: Path, enabled_modules: list[str]) -> dict:
    if not log_path.is_file():
        return {
            "module": None,
            "completed_modules": [],
            "percent": 0.0,
            "done": False,
        }

    text = log_path.read_text(encoding="utf-8", errors="ignore")

    completed = [m for m in enabled_modules if f"[Module: {m}] runtime:" in text]
    current = None
    for m in enabled_modules:
        if f"[Module: {m}] start" in text and f"[Module: {m}] runtime:" not in text:
            current = m

    done = "end time:" in text
    total = max(1, len(enabled_modules))
    percent = min(100.0, round(100.0 * len(completed) / total, 1))
    if done:
        percent = 100.0

    return {
        "module": current,
        "completed_modules": completed,
        "percent": percent,
        "done": done,
    }
