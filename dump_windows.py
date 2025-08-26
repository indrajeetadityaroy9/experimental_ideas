import Quartz
from datetime import datetime

COLS = [
    ("WindowID", "kCGWindowNumber", 8),
    ("PID", "kCGWindowOwnerPID", 6),
    ("App", "kCGWindowOwnerName", 18),
    ("Title", "kCGWindowName", 32),
    ("Bounds", "kCGWindowBounds", 25),
]

def clip(text: str, width: int) -> str:
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"

def bounds_to_str(bounds_dict) -> str:
    if not bounds_dict:
        return ""
    x = int(bounds_dict.get("X", 0))
    y = int(bounds_dict.get("Y", 0))
    w = int(bounds_dict.get("Width", 0))
    h = int(bounds_dict.get("Height", 0))
    return f"{x},{y}→{w}×{h}"

def main() -> None:
    info_arr = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
    )

    windows = list(info_arr)
    windows.sort(key=lambda w: w.get("kCGWindowLayer", 0))

    header = " | ".join(f"{h:<{w}}" for h, _, w in COLS)
    print(header)
    print("-" * len(header))

    for win in windows:
        row = []
        for hdr, key, width in COLS:
            val = win.get(key, "")
            if key == "kCGWindowBounds":
                val = bounds_to_str(val)
            row.append(f"{clip(val, width):<{width}}")

    print(
        f"\n{len(windows)} windows — generated {datetime.now():%Y-%m-%d %H:%M:%S}"
    )

if __name__ == "__main__":
    main()
