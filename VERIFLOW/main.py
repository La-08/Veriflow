import os
import os
from VERIFLOW.llm.deepseek_client import get_llm_response
from VERIFLOW.engines import engine_a, engine_b, engine_c
from VERIFLOW.aggregator.aggregator import aggregate
from VERIFLOW.reporting.report_generator import generate_report, write_report

# preprocessing helpers
from VERIFLOW.preprocessing.cleaner import _norm, _strip_md
from VERIFLOW.preprocessing.sentence_splitter import _split_sentences
from VERIFLOW.preprocessing.claim_extractor import _atomic_claims
from VERIFLOW.preprocessing.entity_extractor import extract_entities


def main():
    question = input("Question: ")
    response = get_llm_response(question)

    # run preprocessing pipeline on the LLM response
    cleaned = _strip_md(_norm(response))
    sentences = _split_sentences(cleaned)
    claims = []
    for s in sentences:
        claims.extend(_atomic_claims(s))
    entities = extract_entities(cleaned)

    prepared = {
        "raw": response,
        "cleaned": cleaned,
        "sentences": sentences,
        "claims": claims,
        "entities": entities,
    }

    # pass the cleaned text as engine input (engines expect text input)
    engine_input = prepared["cleaned"]

    a = engine_a.run_engine_a(engine_input)
    b = engine_b.run_engine_b(engine_input)
    c = engine_c.run_engine_c(engine_input)

    final = aggregate(a, b, c)
    report = generate_report(question, response, a, b, c, final)

    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report_main.json")
    write_report(path, report)
    print("Report written to:", path)


if __name__ == "__main__":
    main()
