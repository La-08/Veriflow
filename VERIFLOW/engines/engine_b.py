from typing import Dict, Any, List, Tuple
import math

from VERIFLOW import shared
from VERIFLOW.preprocessing.preprocessor import _split_sentences, _norm

import numpy as np


def run_engine_b(llm_response: str) -> Dict[str, Any]:
    sentences = _split_sentences(llm_response)

    # dedup
    seen, unique = set(), []
    for s in sentences:
        k = _norm(s).lower()
        if k not in seen:
            seen.add(k)
            unique.append(s)
    sentences = unique

    if len(sentences) < 2:
        return {"engine":"B","engine_name":"Internal Consistency","weight":0.25,
                "score":0.0,"verdict":"consistent","pairs_checked":0,
                "contradictions_found":0,"diagnostics":[]}

    shared.init_models()
    if shared.SBERT is None or shared.st_util is None:
        return {"engine":"B","engine_name":"Internal Consistency","weight":0.25,
                "score":0.0,"verdict":"consistent","pairs_checked":0,
                "contradictions_found":0,"diagnostics":[]}

    emb = shared.SBERT.encode(sentences, convert_to_tensor=True,
                              normalize_embeddings=True, show_progress_bar=False)
    sim = shared.st_util.cos_sim(emb, emb)
    n = len(sentences)

    upper = [float(sim[i][j]) for i in range(n) for j in range(i+1, n)]
    g_mean, g_std = float(np.mean(upper)), float(np.std(upper))
    gate = max(0.52, min(0.92, g_mean + 1.25 * g_std + (0.05 if g_std < 0.06 else 0)))

    candidates: List[Tuple[int,int,float]] = []
    for i in range(n):
        row = [float(sim[i][j]) for j in range(n) if j != i]
        row_gate = max(gate, float(np.percentile(np.array(row, dtype=np.float32), 85.0)))
        scored = sorted([(j, float(sim[i][j])) for j in range(i+1, n)
                         if float(sim[i][j]) >= row_gate], key=lambda x: x[1], reverse=True)
        candidates.extend((i, j, s) for j, s in scored[:4])

    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:300]
    pairs = [(sentences[i], sentences[j], s) for i, j, s in candidates]

    diags = []
    nli_pipe = shared.NLI_PIPE
    preds = []
    if pairs and nli_pipe is not None:
        try:
            payload = [{"text": a, "text_pair": b} for a, b, _ in pairs]
            preds = nli_pipe(payload, batch_size=24, truncation=True, top_k=None)
        except Exception:
            preds = []

    for idx, (a, b, s) in enumerate(pairs):
            p = preds[idx] if idx < len(preds) else None
            c_ab = 0.0
            lbl = ""
            conf = 0.0
            if p:
                if isinstance(p, list):
                    best = max(p, key=lambda x: float(x.get("score",0)))
                    lbl = best.get("label", "")
                    conf = float(best.get("score", 0.0))
                    c_ab = max((float(x.get("score",0)) for x in p if "contradiction" in str(x.get("label","")).lower()), default=0.0)
                else:
                    lbl = p.get("label", "")
                    conf = float(p.get("score", 0.0))
                    if "contradiction" in str(lbl).lower():
                        c_ab = float(p.get("score", 0.0))

            c_ba = 0.0
            if "contradiction" in str(lbl).lower() or float(conf) < 0.85:
                if nli_pipe is not None:
                    try:
                        r = nli_pipe({"text": b, "text_pair": a}, truncation=True, top_k=None)
                        if isinstance(r, list):
                            c_ba = max((float(x.get("score",0)) for x in r if "contradiction" in str(x.get("label","")).lower()), default=0.0)
                    except Exception:
                        pass

            base = max(c_ab, c_ba)
            diags.append({"sentence_a":a,"sentence_b":b,"similarity":s,
                           "contradiction_score":base})

    contra_scores = [d["contradiction_score"] for d in diags if d["contradiction_score"] > 0.20]

    if not contra_scores:
        final = 0.0
    else:
        probs = [min(1.0, max(0.0, x)) for x in contra_scores]
        log_none = sum(math.log(max(1e-12, 1.0 - p)) for p in probs)
        p_any = 1.0 - math.exp(log_none)
        density = max(3.0, len(sentences) * 0.10)
        intensity = min(1.0, sum(probs) / density)
        final = min(1.0, max(0.0, (0.60*p_any + 0.40*intensity) * 0.25))

    verdict = "contradicted" if final > 0.15 else "consistent"
    return {
        "engine":"B","engine_name":"Internal Consistency","weight":0.25,
        "score":round(final,4),"verdict":verdict,
        "pairs_checked":len(pairs),"contradictions_found":len(contra_scores),
        "diagnostics":[
            {"sentence_a":d["sentence_a"][:120],"sentence_b":d["sentence_b"][:120],
             "similarity":round(d["similarity"],4),
             "contradiction_score":round(d["contradiction_score"],4)}
            for d in diags if d["contradiction_score"] > 0.20
        ][:5],
    }
