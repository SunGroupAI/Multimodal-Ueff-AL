"""Small standalone plot-area alignment check for the distributed figure script."""
import json
from pathlib import Path


def require_matplotlib_panel_alignment(fig, axes=None, panel_ids=None, row_groups=None,
                                      column_groups=None, json_out=None, tolerance_pt=1.5,
                                      **kwargs):
    fig.canvas.draw()
    axes = list(axes if axes is not None else fig.axes)
    ids = list(panel_ids if panel_ids is not None else map(str, range(len(axes))))
    rects = {}
    width, height = fig.get_size_inches() * 72
    for label, ax in zip(ids, axes):
        p = ax.get_position()
        rects[label] = [p.x0 * width, p.y0 * height, p.x1 * width, p.y1 * height]
    failures = []
    for groups, coords in [(row_groups or [], [1, 3]), (column_groups or [], [0, 2])]:
        for group in groups:
            for coord in coords:
                vals = [rects[label][coord] for label in group]
                if max(vals) - min(vals) > tolerance_pt:
                    failures.append({"group": group, "coordinate": coord})
    report = {"scope": "plot-area edges only", "rectangles_pt": rects,
              "tolerance_pt": tolerance_pt, "failures": failures,
              "status": "PASS" if not failures else "FAIL"}
    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        raise ValueError(f"Panel alignment failed: {failures}")
