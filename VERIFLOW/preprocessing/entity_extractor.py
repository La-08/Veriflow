from typing import List, Tuple

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None


def extract_entities(text: str) -> List[Tuple[str, str]]:
    if not _NLP:
        return []
    doc = _NLP(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
