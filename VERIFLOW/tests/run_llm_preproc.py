import os
import sys
from pathlib import Path

# load .env manually
env_path = Path(__file__).resolve().parents[2] / '.env'
if env_path.exists():
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

from VERIFLOW.llm.deepseek_client import get_llm_response
from VERIFLOW.preprocessing.cleaner import _norm, _strip_md
from VERIFLOW.preprocessing.sentence_splitter import _split_sentences
from VERIFLOW.preprocessing.claim_extractor import _atomic_claims
from VERIFLOW.preprocessing.entity_extractor import extract_entities


def run_once(question: str):
    print('=== QUESTION ===')
    print(question)
    print('\n=== CALLING LLM (debug) ===')
    try:
        resp = get_llm_response(question, debug=True)
    except Exception as e:
        print('LLM call failed:', e)
        sys.exit(1)
    print('\n=== LLM RESPONSE ===')
    print(resp)

    print('\n=== PREPROCESSING ===')
    cleaned = _strip_md(_norm(resp))
    print('CLEANED:', cleaned)
    sents = _split_sentences(cleaned)
    print('SENTENCES:', sents)
    claims = []
    for s in sents:
        claims.extend(_atomic_claims(s))
    print('CLAIMS:', claims)
    entities = extract_entities(cleaned)
    print('ENTITIES:', entities)


if __name__ == '__main__':
    q = 'Tell me about the Eiffel Tower.'
    run_once(q)
