import os
import sys

# make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from VERIFLOW.engines.engine_a import run_engine_a
from VERIFLOW.engines.engine_b import run_engine_b
from VERIFLOW.engines.engine_c import run_engine_c
from VERIFLOW.aggregator.aggregator import aggregate


def main():
    print("VERIFLOW live demo — enter a question (or blank to exit)")
    while True:
        q = input("Question> ").strip()
        if not q:
            break

        # Placeholder DeepSeek call — replace with your retrieval/LLM call
        # For demo purposes we treat the question itself as the LLM response.
        llm_response = q

        ra = run_engine_a(llm_response)
        rb = run_engine_b(llm_response)
        rc = run_engine_c(llm_response)

        agg = aggregate(ra, rb, rc)

        print("\nFinal aggregate:", agg)


if __name__ == "__main__":
    main()
