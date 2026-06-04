VERIFLOW

VERIFLOW is a framework for evaluating the reliability of AI-generated responses. Large language models can produce answers that sound convincing but are factually incorrect, contradictory, or unsupported. VERIFLOW analyzes responses from multiple perspectives and combines the results into a final reliability score.

Why I Built This

Modern LLM outputs often sound confident even when they are wrong. That makes hallucinations difficult to detect, especially for users who do not already know the correct answer.

VERIFLOW introduces a layered verification pipeline that examines:


factual accuracy
internal consistency
confidence and bias patterns

How It Works

User Question

↓ 

DeepSeek

↓ 

AI Response

↓

Preprocessing 

↓ 

Engine A — Fact Verification

↓ 

Engine B — Consistency Analysis

↓ 

Engine C — Confidence & Bias Analysis ↓ Aggregator 

↓ 

Reliability Score

Components

Engine A – Fact Verification
Checks whether factual claims in the response can be supported by external evidence.

Tools used:

Wikipedia API
Sentence-BERT
Natural Language Inference (NLI)
Engine B – Internal Consistency Analysis
Looks for contradictions within the same response.

Example:

"Water boils at 100°C."
"Water boils at 90°C."
The response is internally inconsistent even if it appears fluent.

Engine C – Confidence & Bias Analysis
Analyzes linguistic patterns that may indicate unreliable responses.

Examples include:

excessive certainty
unsupported confidence
absolutist language
linguistic bias
Aggregator
Combines outputs from all engines and generates a final reliability score.

Technologies Used
Python
DeepSeek (via OpenRouter)
spaCy
Sentence Transformers
Hugging Face Transformers
PyTorch
Wikipedia API
Sample Output
Question: Who is the President of India?

Engine A: Verified 
Engine B: Consistent
Engine C: Low Risk

Final Reliability Score: 0.12 Risk Level: Reliable

Future Improvements
Support for additional evidence sources
Web-based dashboard
Explainable verification reports
Support for multiple LLM providers
Enhanced hallucination classification

"Trust, but verify. Evaluating AI reliability through multi-layer verification."
