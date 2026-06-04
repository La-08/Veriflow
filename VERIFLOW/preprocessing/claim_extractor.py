import re
from typing import List


def _atomic_claims(text: str) -> List[str]:
    text = text.strip().rstrip('.')
    segs = re.split(r", so |, because | and |, not a | but |;", text)
    claims = [s.strip().rstrip('.') for s in segs if len(s.split()) > 2]
    if not claims:
        return [text]
    return claims
