import json
from typing import Dict, Any


def generate_report(question: str, response: str, a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, Any]:
    report = {
        "question": question,
        "response": response,
        "engine_a": a,
        "engine_b": b,
        "engine_c": c,
        "final_score": final.get("final_score" if isinstance(final, dict) else "score", None),
    }
    return report


def write_report(path: str, report: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
