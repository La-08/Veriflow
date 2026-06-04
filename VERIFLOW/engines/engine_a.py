from typing import Any, Dict, List, Optional
import re
import requests

from VERIFLOW import shared
from VERIFLOW.preprocessing.preprocessor import _strip_md, _simplify, _split_sentences, _atomic_claims, _keywords


# _retrieve_evidence and run_engine_a moved here

def _wikipedia_search(query: str, results: int = 3) -> List[str]:
    try:
        import wikipedia
        return wikipedia.search(query, results=results)
    except Exception:
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            headers = {"User-Agent": "VERIFLOW/1.0 (https://github.com/)"}
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": results,
            }
            resp = requests.get(api_url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [item["title"] for item in data.get("query", {}).get("search", [])]
        except Exception:
            return []


def _retrieve_page_content(title: str) -> Optional[str]:
    page = None
    try:
        import wikipedia
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
        except Exception:
            page = None
        if page is not None:
            return (page.summary or "") + "\n" + (page.content or "")[:4000]
    except Exception:
        page = None

    try:
        encoded = requests.utils.quote(title, safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        headers = {"User-Agent": "VERIFLOW/1.0 (https://github.com/)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("extract", "") or "")[:4000]
    except Exception:
        return None


def _retrieve_evidence(claim: str) -> List[str]:
    shared.init_models(silent=False)

    clean = _strip_md(claim)
    simplified = _simplify(clean)

    all_sents: List[str] = []
    titles = _wikipedia_search(simplified, results=3)

    if not titles:
        candidate_titles = []
        if shared._NLP:
            candidate_titles.extend(
                ent.text for ent in shared._NLP(clean).ents
                if ent.label_ in ("ORG", "GPE", "LOC", "PERSON", "WORK_OF_ART", "EVENT")
            )
        candidate_titles.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", clean))
        keywords = _keywords(simplified)
        if keywords and keywords not in candidate_titles:
            candidate_titles.append(keywords)
        direct_query = " ".join(simplified.split()[:6])
        if direct_query and direct_query not in candidate_titles:
            candidate_titles.append(direct_query)
        if simplified and simplified not in candidate_titles:
            candidate_titles.append(simplified)

        for candidate in dict.fromkeys(candidate_titles):
            if not candidate:
                continue
            content = _retrieve_page_content(candidate)
            if content:
                titles.append(candidate)
                break

    for title in titles:
        content = _retrieve_page_content(title)
        if not content:
            continue
        sents = _split_sentences(content)
        if sents:
            all_sents.extend(sents)

    if not all_sents and simplified:
        fallback_query = " ".join(simplified.split()[:6])
        fallback_titles = _wikipedia_search(fallback_query, results=2)
        for title in fallback_titles:
            content = _retrieve_page_content(title)
            if content:
                sents = _split_sentences(content)
                all_sents.extend(sents)
                if sents:
                    pass

    if not all_sents:
        return []

    chunks = [" ".join(all_sents[i:i+3]) for i in range(0, len(all_sents), 2)]
    seen, unique = set(), []
    for s in chunks:
        k = s.strip().lower()
        if k not in seen:
            seen.add(k)
            unique.append(s)

    if shared.SBERT is not None and shared.cosine_similarity is not None:
        claim_emb = shared.SBERT.encode([simplified])
        sent_embs = shared.SBERT.encode(unique)
        scores = shared.cosine_similarity(claim_emb, sent_embs)[0]
        ranked = sorted(zip(scores, unique), reverse=True)
        filtered = [s for sc, s in ranked if sc >= 0.25]
        result = (filtered or [s for _, s in ranked])[:7]
        return result

    return unique[:7]
    return unique[:7]


def run_engine_a(llm_response: str) -> Dict[str, Any]:
    claims = _atomic_claims(llm_response)
    claim_scores: List[Optional[float]] = []
    claim_details: List[Dict] = []

    for claim in claims:
        evidence = _retrieve_evidence(claim)
        if not evidence:
            claim_scores.append(None)
            claim_details.append({"claim": claim, "status": "unverifiable", "score": None})
            continue

        # If NLI model exists, run; otherwise mark unverifiable
        if shared.NLI_CROSS is None:
            claim_scores.append(None)
            claim_details.append({"claim": claim, "status": "unverifiable", "score": None})
            continue

        nli_claim = _simplify(claim)
        pairs = [[ev, nli_claim] for ev in evidence]
        try:
            logits = shared.NLI_CROSS.predict(pairs)
        except Exception:
            logits = None
        import numpy as np
        import torch
        import torch.nn.functional as F
        if logits is None or (isinstance(logits, np.ndarray) and logits.size == 0):
            claim_scores.append(None)
            claim_details.append({"claim": claim, "status": "unverifiable", "score": None})
            continue

        probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        label_map = {label.lower(): idx for idx, label in shared.NLI_CROSS.config.id2label.items()}
        ent_idx = label_map.get("entailment", label_map.get("ent", 2))
        con_idx = label_map.get("contradiction", label_map.get("contra", 0))

        ent = float(probs[:, ent_idx].max()) if ent_idx < probs.shape[1] else 0.0
        con = float(probs[:, con_idx].max()) if con_idx < probs.shape[1] else 0.0

        if ent > 0.45 and con < 0.40:
            score = ent; status = "verified"
        elif con > 0.50:
            score = 0.0; status = "contradicted"
        else:
            score = None; status = "unverifiable"

        claim_scores.append(score)
        claim_details.append({"claim": claim, "status": status,
                               "score": score, "evidence_count": len(evidence)})

    import numpy as np
    valid = [s for s in claim_scores if s is not None]
    n_unv = claim_scores.count(None)

    if not valid:
        final_score = 0.5; verdict = "unverifiable"
    else:
        avg = float(np.mean(valid))
        final_score = 1.0 - avg
        verdict = "verified" if avg > 0.70 else ("uncertain" if avg > 0.40 else "hallucinated")

    return {
        "engine": "A", "engine_name": "Fact Verification",
        "weight": 0.40, "score": round(final_score, 4), "verdict": verdict,
        "claim_details": claim_details,
        "n_verified": sum(1 for d in claim_details if d["status"] == "verified"),
        "n_contradicted": sum(1 for d in claim_details if d["status"] == "contradicted"),
        "n_unverifiable": n_unv,
    }
