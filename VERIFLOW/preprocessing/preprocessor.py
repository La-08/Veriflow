import re
import unicodedata
from functools import lru_cache
from typing import List

from VERIFLOW import shared

@lru_cache(maxsize=4096)
def _norm(text: str) -> str:
    try:
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        pass
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return shared.WHITESPACE.sub(" ", text).strip()


def _strip_md(text: str) -> str:
    # lightweight markdown stripping
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text.strip()


def _keywords(claim: str) -> str:
    if shared._NLP:
        doc = shared._NLP(claim)
        kw = [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN", "VERB")
              and t.lemma_.lower() not in shared.STOPWORDS and len(t.text) > 2 and not t.is_punct]
        return " ".join(kw[:5])
    return " ".join(w for w in claim.split() if w.lower() not in shared.STOPWORDS and len(w) > 2)[:50]


def _simplify(claim: str) -> str:
    s = re.sub(r",\s*(serving|having|who|which|where|when|currently).*$", "", claim, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


def _split_sentences(text: str) -> List[str]:
    text = _strip_md(_norm(text))
    if shared._NLP:
        return [s.text.strip() for s in shared._NLP(text).sents if s.text.strip() and len(s.text.split()) > 3]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip() and len(s.split()) > 3]


def _atomic_claims(text: str) -> List[str]:
    text = _strip_md(text)
    segs = re.split(r", so |, because | and |, not a | but ", text)
    claims = [s.strip().rstrip(".") for s in segs if len(s.split()) > 2]
    if not claims:
        claims = _split_sentences(text)
    return claims
