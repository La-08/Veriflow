import io
import os
import re
import warnings
import unicodedata
from contextlib import redirect_stdout, redirect_stderr
from functools import lru_cache

# Environment
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Model singletons (initialised lazily)
SBERT = None
NLI_CROSS = None
NLI_PIPE = None
ZERO_SHOT = None
st_util = None
cosine_similarity = None
_NLP = None  # optional spaCy NLP pipeline


def init_models(silent: bool = True):
    global SBERT, NLI_CROSS, NLI_PIPE, ZERO_SHOT, st_util, cosine_similarity, _NLP
    if SBERT is not None and NLI_CROSS is not None:
        return

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            from sentence_transformers import CrossEncoder, SentenceTransformer, util as st_util_mod
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            from transformers import pipeline as hf_pipeline
            try:
                import spacy
                _NLP = spacy.load("en_core_web_sm")
            except Exception:
                _NLP = None

            if silent:
                stdout = open(os.devnull, "w")
                stderr = open(os.devnull, "w")
                redirect_ctx = redirect_stdout(stdout)
                stderr_ctx = redirect_stderr(stderr)
            else:
                redirect_ctx = redirect_stdout(io.StringIO())
                stderr_ctx = redirect_stderr(io.StringIO())

            with redirect_ctx, stderr_ctx:
                SBERT = SentenceTransformer("all-MiniLM-L6-v2")
                NLI_CROSS = CrossEncoder("cross-encoder/nli-deberta-v3-base")
                device = 0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1
                try:
                    NLI_PIPE = hf_pipeline("text-classification",
                                          model="cross-encoder/nli-deberta-v3-base",
                                          device=device, truncation=True)
                except Exception:
                    NLI_PIPE = None
                try:
                    ZERO_SHOT = hf_pipeline("zero-shot-classification",
                                            model="facebook/bart-large-mnli",
                                            device=device)
                except Exception:
                    ZERO_SHOT = None

            st_util = st_util_mod
            cosine_similarity = cos_sim
    except Exception:
        SBERT = None
        NLI_CROSS = None
        NLI_PIPE = None
        ZERO_SHOT = None
        st_util = None
        cosine_similarity = None
        _NLP = None


# Shared constants & text utilities
STOPWORDS = {
    "no","a","an","the","has","have","is","are","was","were","yet","never",
    "not","been","any","ever","still","also","just","that","this","with",
    "from","for","its","it","be","by","to","of","in","on","at","as","or",
    "and","but","so","if","do",
}
WHITESPACE = re.compile(r"\s+")

LOADED_LANGUAGE = [
    "radical","extremist","regime","thug","hero","terrorist","freedom fighter",
    "corrupt","dishonest","propaganda","fake","hoax","lie","brainwash",
    "manipulate","conspiracy","agenda","biased","toxic","brilliant","genius",
    "disastrous","catastrophic","revolutionary","epic","unprecedented",
]
ONE_SIDED_HEDGES = [
    "clearly","obviously","everyone knows","undeniably","without a doubt",
    "it is a fact","it goes without saying","of course","needless to say",
    "it is obvious","plain to see",
]
FRAMING_CUES = {
    r"\bfailed\b":"negative framing", r"\bso-called\b":"delegitimization",
    r"\ballege[sd]?\b":"unverified attribution", r"\bclaim[sed]*\b":"skeptical framing",
    r"\bpurportedly\b":"skeptical framing", r"\bextremist[s]?\b":"loaded label",
}
CERTAINTY_MARKERS = [
    "definitely","certainly","absolutely","undoubtedly","unquestionably",
    "without doubt","100%","always","never","everyone","no one","all",
    "every","impossible","guaranteed","proven fact","scientific consensus",
    "it is a fact that","it is well known that","there is no doubt",
]
HEDGING_MARKERS = [
    "may","might","could","possibly","perhaps","likely","probably","appears",
    "seems","suggests","indicates","approximately","around","estimated",
    "according to","it is believed","some evidence","often","generally",
    "typically","in many cases","tends to","as of","researchers say",
]

ENGINE_WEIGHTS = {"A": 0.40, "B": 0.25, "C": 0.35}
RISK_THRESHOLDS = {"HIGH": 0.65, "MODERATE": 0.40, "LOW": 0.20}

# Expose utilities
__all__ = [
    "init_models",
    "SBERT","NLI_CROSS","NLI_PIPE","ZERO_SHOT","st_util","cosine_similarity","_NLP",
    "STOPWORDS","WHITESPACE","LOADED_LANGUAGE","ONE_SIDED_HEDGES","FRAMING_CUES",
    "CERTAINTY_MARKERS","HEDGING_MARKERS","ENGINE_WEIGHTS","RISK_THRESHOLDS",
]
