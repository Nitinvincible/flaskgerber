import re
from typing import List, Tuple
from app import DrillHole

def parse_advanced_excellon(filepath: str) -> List[DrillHole]:
    """
    Parse more advanced Excellon drill files, supporting both PTH and NPTH, and various header formats.
    Returns a list of DrillHole objects.
    """
    holes = []
    tools = {}
    units = 'mm'
    current_tool = None

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Units
        if 'INCH' in line.upper():
            units = 'inch'
        elif 'METRIC' in line.upper():
            units = 'mm'

        # Tool definition (T01C0.600)
        m = re.match(r'^T(\d+)[C|F]([\d\.]+)', line, re.IGNORECASE)
        if m:
            tool_id = f"T{m.group(1).zfill(2)}"
            diameter = float(m.group(2))
            tools[tool_id] = diameter
            continue

        # Tool selection (T01)
        if re.match(r'^T\d+$', line):
            current_tool = f"T{line[1:].zfill(2)}"
            continue

        # Coordinates (XnnnYnnn)
        m = re.match(r'X([-+]?\d+\.?\d*)Y([-+]?\d+\.?\d*)', line)
        if m and current_tool:
            x = float(m.group(1))
            y = float(m.group(2))
            diameter = tools.get(current_tool, 0.1)
            if units == 'inch':
                x *= 25.4
                y *= 25.4
                diameter *= 25.4
            holes.append(DrillHole(x, y, diameter))
            continue

        # NPTH/PTH detection (optional, for future extension)
        # Could parse M15/M16 blocks for NPTH/PTH, or look for "NPTH" in comments

    return holes

def is_excellon_drill_file(filepath: str) -> bool:
    """
    Heuristically determine if a file is an Excellon drill file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(2048)
    except Exception:
        return False

    # Look for Excellon header or tool definitions
    if 'M48' in content or re.search(r'T\d+C[\d\.]+', content):
        return True
    if re.search(r'X[-+]?\d+Y[-+]?\d+', content):
        return True
    return False
