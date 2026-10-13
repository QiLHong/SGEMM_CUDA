"""Strict Compute Sanitizer log checks shared by experiment drivers/reports."""
import re


def sanitizer_passed(content, tool):
    """Require a numerical PASS and exactly one tool-specific zero summary.

    Substring checks for '0 errors' also match '10 errors', and race warnings
    can coexist with zero errors. Neither is a clean sanitizer result.
    """
    if not re.search(r"^RESULT,[^\n]*,PASS\s*$", content, re.MULTILINE):
        return False
    if tool == "racecheck":
        summaries = re.findall(
            r"^=+\s+RACECHECK SUMMARY:\s+(\d+) hazards? displayed "
            r"\((\d+) errors?, (\d+) warnings?\)\s*$", content, re.MULTILINE)
        return summaries == [("0", "0", "0")]
    if tool not in ("memcheck", "synccheck"):
        raise ValueError(f"Unsupported sanitizer: {tool}")
    summaries = re.findall(r"^=+\s+ERROR SUMMARY:\s+(\d+) errors?\s*$",
                           content, re.MULTILINE)
    return summaries == ["0"]
