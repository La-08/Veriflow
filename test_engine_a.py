#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VERIFLOW.engines.engine_a import run_engine_a

# Test with a simple claim
claim = "The Eiffel Tower is located in Paris, France."
result = run_engine_a(claim)
print("Engine A Result:")
print(result)
