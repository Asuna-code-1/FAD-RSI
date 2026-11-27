"""Table 1's equally weighted five-task accuracy, in percentage points."""
import math

PAPER_MATH_KEYS = tuple(
    f"test-legacy-core/{task}/acc/mean@16"
    for task in ("aime24", "aime25", "amc23", "AMO-Bench", "hmmt_feb_2026")
)
PAPER_AVERAGE_KEY = "test-legacy-core/Average"


def paper_math_average(metrics):
    """Return None for incomplete evaluations; never combine different steps."""
    if not all(key in metrics for key in PAPER_MATH_KEYS):
        return None
    values = [float(metrics[key]) for key in PAPER_MATH_KEYS]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        return None
    return 100.0 * math.fsum(values) / len(values)
