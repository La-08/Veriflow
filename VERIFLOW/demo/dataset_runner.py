import os
import sys
import json

# make project root importable so demos work when run from the VERIFLOW folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from VERIFLOW.datasets.dataset_loader import load_fever_samples
from VERIFLOW.engines.engine_a import run_engine_a
from VERIFLOW.engines.engine_b import run_engine_b
from VERIFLOW.engines.engine_c import run_engine_c
from VERIFLOW.aggregator.aggregator import aggregate


def main():
    ds_path = os.path.join(PARENT, "VERIFLOW", "datasets", "fever_dev.jsonl")
    samples = load_fever_samples(ds_path, limit=10)

    out_dir = os.path.join(PARENT, "VERIFLOW", "outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)

    for idx, sample in enumerate(samples):
        response = sample["claim"]
        print("\n====================")
        print("CLAIM:", response)

        ra = run_engine_a(response)
        rb = run_engine_b(response)
        rc = run_engine_c(response)

        final = aggregate(ra, rb, rc)

        print(final)

        # save per-sample JSON report
        fname = f"report_{idx+1}.json"
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
            json.dump({"claim": response, "label": sample.get("label"), "A": ra, "B": rb, "C": rc, "agg": final}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()