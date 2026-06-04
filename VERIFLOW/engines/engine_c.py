import re
from typing import Dict, Any, List

from VERIFLOW import shared


def run_engine_c(llm_response: str) -> Dict[str, Any]:
    shared.init_models()
    text_l = llm_response.lower()

    # C1 — Bias / Framing
    bias_hits: List[str] = []
    for word in shared.LOADED_LANGUAGE:
        if re.search(rf"\b{re.escape(word)}\b", text_l):
            bias_hits.append(f"loaded: '{word}'")
    for phrase in shared.ONE_SIDED_HEDGES:
        if phrase in text_l:
            bias_hits.append(f"one-sided: '{phrase}'")
    for pattern, label in shared.FRAMING_CUES.items():
        if re.search(pattern, text_l):
            bias_hits.append(label)
    lex_score = min(1.0, len(bias_hits) / 6.0)

    zs_score = lex_score
    if shared.ZERO_SHOT:
        try:
            result = shared.ZERO_SHOT(llm_response[:512],
                               candidate_labels=["biased","neutral","factual"],
                               multi_label=False)
            label_scores = dict(zip(result["labels"], result["scores"]))
            zs_score = float(label_scores.get("biased", lex_score))
        except Exception:
            pass
    bias_score = min(1.0, 0.5*lex_score + 0.5*zs_score)

    # C2 — Overconfidence
    certainty_hits = [m for m in shared.CERTAINTY_MARKERS if m in text_l]
    hedging_hits   = [m for m in shared.HEDGING_MARKERS   if m in text_l]
    n_words        = max(len(text_l.split()), 1)
    c_rate = (len(certainty_hits) / n_words) * 100
    h_rate = (len(hedging_hits)   / n_words) * 100
    ratio  = c_rate / (c_rate + h_rate + 1e-9) if (c_rate + h_rate) > 0 else 0.2

    unhedged = 0
    if shared._NLP:
        for sent in shared._NLP(llm_response).sents:
            s_l = sent.text.lower()
            if (not any(h in s_l for h in shared.HEDGING_MARKERS)
                    and any(t.pos_ == "VERB" for t in sent)
                    and any(t.pos_ in ("NOUN","PROPN") for t in sent)):
                unhedged += 1
    overconf_score = min(1.0, ratio + min(0.4, unhedged * 0.05))

    combined_raw = 0.20 * bias_score + 0.15 * overconf_score
    normalised   = combined_raw / 0.35 if 0.35 != 0 else 0.0

    bias_verdict = ("high" if bias_score > 0.55 else
                    "moderate" if bias_score > 0.25 else "low")
    oc_verdict   = ("overconfident" if overconf_score > 0.60 else
                    "somewhat overconfident" if overconf_score > 0.35 else "calibrated")

    return {
        "engine":"C","engine_name":"Linguistic & Rhetorical","weight":0.35,
        "score":round(normalised,4),
        "bias_score":round(bias_score,4),
        "overconfidence_score":round(overconf_score,4),
        "bias_verdict":bias_verdict,
        "overconfidence_verdict":oc_verdict,
        "bias_indicators":bias_hits,
        "certainty_markers":certainty_hits,
        "hedging_markers":hedging_hits,
        "unhedged_factual_sentences":unhedged,
    }
