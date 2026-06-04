import re
from typing import List
from .cleaner import _norm, _strip_md


def _split_sentences(text: str) -> List[str]:
    text = _strip_md(_norm(text))
    # simple sentence split fallback
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
