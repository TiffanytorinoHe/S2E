"""Section-local colour interpolation from the main-figure renderer."""
import numpy as np
def geom_arrays(rows, values, model):
    """Interpolate colour along each section, never across unrelated branches."""
    lines, widths, colors = [], [], []
    sections = {}
    soma_points, soma_values = [], []
    for row in rows:
        if row["is_axon"]:
            continue
        if "soma" in row["section"]:
            points = np.array(row["xyz"])[:, :2]
            soma_points.extend(points.tolist())
            soma_values.append(values[row["index"]])
        else:
            sections.setdefault(row["section"], []).append(row)
    for group in sections.values():
        group.sort(key=lambda r: r["x"])
        x = [r["x"] for r in group]
        val = [values[r["index"]] for r in group]
        for row in group:
            xyz = np.array(row["xyz"])
            length = np.r_[0, np.linalg.norm(np.diff(xyz, axis=0), axis=1).cumsum()]
            if length[-1] <= 1e-9:
                continue
            pos = np.unique(np.r_[length, np.linspace(0, length[-1], max(2, int(length[-1]/2.2)+2))])
            points = np.column_stack([np.interp(pos, length, xyz[:, k]) for k in range(2)])
            mids = (pos[:-1]+pos[1:])/(2*length[-1])
            local_x = row["x"]+(mids-.5)/len(group)
            colors.extend(np.interp(local_x, x, val))
            lines.extend(np.stack([points[:-1], points[1:]], axis=1))
            widths.extend([max(4.95, row["diam"]*2.18)]*(len(points)-1))
    lines = np.asarray(lines)
    soma_points = np.array(soma_points)
    if model == "ca1_20212":
        lines[:, :, 1] *= -1
        soma_points[:, 1] *= -1
    return lines, np.array(widths), np.array(colors), soma_points, max(soma_values)
