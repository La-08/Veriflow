from typing import Dict, Any
from VERIFLOW.shared import ENGINE_WEIGHTS, RISK_THRESHOLDS


def aggregate(ra: Dict, rb: Dict, rc: Dict) -> Dict[str, Any]:
    scores = {"A": ra["score"], "B": rb["score"], "C": rc["score"]}
    weighted_avg = sum(ENGINE_WEIGHTS[k] * v for k, v in scores.items())
    p_none = 1.0
    for k, v in scores.items():
        p_none *= (1.0 - ENGINE_WEIGHTS[k] * min(1.0, v))
    p_any = 1.0 - p_none
    final = min(1.0, 0.60 * p_any + 0.40 * weighted_avg)

    if   final >= RISK_THRESHOLDS["HIGH"]:     tier, emoji = "HIGH",     "🔴"
    elif final >= RISK_THRESHOLDS["MODERATE"]: tier, emoji = "MODERATE", "🟡"
    elif final >= RISK_THRESHOLDS["LOW"]:      tier, emoji = "LOW",      "🟢"
    else:                                      tier, emoji = "MINIMAL",  "✅"

    contribs = {k: ENGINE_WEIGHTS[k] * v for k, v in scores.items()}
    return {
        "final_score": round(final, 4),
        "weighted_avg": round(weighted_avg, 4),
        "probabilistic_union": round(p_any, 4),
        "risk_tier": tier,
        "risk_emoji": emoji,
        "engine_scores": scores,
        "engine_contributions": {k: round(v, 4) for k, v in contribs.items()},
        "dominant_engine": max(contribs, key=contribs.get),
    }
