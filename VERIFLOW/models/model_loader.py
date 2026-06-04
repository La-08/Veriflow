"""Central model loader. Import this module and call `init_models()` once on startup.
Expose: nlp, sbert, nli_model, zero_shot, util objects.
"""
from typing import Optional

nlp = None
sbert = None
nli_model = None
zero_shot = None
st_util = None


def init_models(silent: bool = True):
    global nlp, sbert, nli_model, zero_shot, st_util
    if sbert is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer, util as st_util_mod
        from transformers import pipeline as hf_pipeline
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None

        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        st_util = st_util_mod

        try:
            nli_model = hf_pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")
        except Exception:
            nli_model = None

        try:
            zero_shot = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception:
            zero_shot = None
    except Exception:
        nlp = sbert = nli_model = zero_shot = st_util = None


__all__ = ["init_models", "nlp", "sbert", "nli_model", "zero_shot", "st_util"]
