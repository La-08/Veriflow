import re
import unicodedata
from functools import lru_cache

WHITESPACE = re.compile(r"\s+")


@lru_cache(maxsize=4096)
def _norm(text: str) -> str:
    try:
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        pass
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return WHITESPACE.sub(" ", text).strip()


def _strip_md(text: str) -> str:
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text.strip()


def _simplify(claim: str) -> str:
    s = re.sub(r",\s*(serving|having|who|which|where|when|currently).*$", "", claim, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()
