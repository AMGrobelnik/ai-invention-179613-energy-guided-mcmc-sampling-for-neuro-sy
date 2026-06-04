# Math Specification for Energy Function and ConceptNet

## Summary

This research artifact provides comprehensive technical specifications for implementing a composite energy function E = w1*E_text + w2*E_onto + w3*E_cohere to guide Metropolis-Hastings MCMC sampling for neuro-symbolic text-to-FOL translation. The research covers: (1) E_text component using Sentence-BERT with all-MiniLM-L6-v2 (384-dim, 5x faster) or all-mpnet-base-v2 (768-dim, best quality) with cosine similarity formula E_text = 1 - cos_sim(SBERT(text), SBERT(program)). (2) E_onto component using ConceptNet API (http://api.conceptnet.io) with 18 relation types including IsA, PartOf, UsedFor, etc., detecting ontological violations when no supporting edges exist. (3) E_cohere component using Datalog with stratified negation semantics, detecting contradictions when both P and not P are entailed; Python libraries evaluated include pyDatalog (unmaintained, consider IDP-Z3 instead) and pyswip (Python-SWI-Prolog bridge). (4) MCMC implementation with Metropolis-Hastings algorithm using acceptance criterion min(1, exp(-delta/T)), proposal distributions for logic programs (add/remove/swap clauses), and temperature scheduling. (5) Prolog integration via pyswip requiring SWI-Prolog installation. All components include exact mathematical formulations, API endpoints, code snippets, and implementation recommendations with pros/cons analysis. The complete energy function is E(P, T, Q) = w1*E_text(P, T) + w2*E_onto(P) + w3*E_cohere(P, Q) with 

## Research Findings

This research provides mathematical specifications and implementation guidance for the composite energy function E = w1*E_text + w2*E_onto + w3*E_cohere used in energy-guided MCMC sampling for neuro-symbolic text-to-FOL translation [1, 2, 3].

## E_text Component (Sentence-BERT)

The E_text component measures semantic fidelity between the input text T and the candidate logic program P using Sentence-BERT embeddings [1]. The mathematical formulation is:

E_text = 1 - cos_sim(SBERT(T), SBERT(P))

where cos_sim(a, b) = (a · b) / (||a|| * ||b||) [1].

For implementation, two models are recommended:
- all-MiniLM-L6-v2: 384-dimensional embeddings, 5x faster than mpnet, suitable for speed-critical applications [2]
- all-mpnet-base-v2: 768-dimensional embeddings, best quality for accuracy-critical applications [2]

The Python implementation uses sentence-transformers library:
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
emb1 = model.encode('text')
emb2 = model.encode('program')
similarity = util.cos_sim(emb1, emb2)
e_text = 1 - similarity.item()
```

## E_onto Component (ConceptNet)

The E_onto component measures ontological consistency by querying ConceptNet for plausible relations between entities in the logic program [3]. The energy is computed as:

E_onto = count of relation-type violations

where a violation occurs when ConceptNet has no edge supporting a predicate's semantic plausibility [3].

ConceptNet API specification:
- Base URL: http://api.conceptnet.io (Note: API returned 502 errors during research, may be temporarily down) [3]
- Query endpoint: /query?start=/c/en/{concept1}&end=/c/en/{concept2}&rel=/r/{relation}
- Concept lookup: /c/en/{concept}

Supported relation types include: /r/IsA, /r/PartOf, /r/HasA, /r/UsedFor, /r/CapableOf, /r/AtLocation, /r/Causes, /r/HasSubevent, /r/MotivatedByGoal, /r/CausesDesire, /r/Desires, /r/HasProperty, etc. [3]

Implementation approach:
```python
import requests
response = requests.get(f'http://api.conceptnet.io/query?start=/c/en/{subj}&end=/c/en/{obj}&rel=/r/{rel}')
data = response.json()
edges = data.get('edges', [])
violation = len(edges) == 0  # No supporting evidence
```

Rate limiting: Not officially documented; recommend ~1 request/second with caching [3].

## E_cohere Component (Datalog)

The E_cohere component measures logical coherence using Datalog with stratified negation semantics [4]. The energy formulation:

E_cohere = w_query * failed_queries + w_contra * contradiction_flag

where:
- failed_queries: count of Datalog queries that fail (return False)
- contradiction_flag: 1 if contradiction detected (both P and not P entailed), 0 otherwise

Datalog semantics are defined via [4]:
- Model-theoretic semantics
- Fixed-point semantics 
- Proof-theoretic semantics

Stratified negation: Negation is only allowed in later strata to avoid inconsistency [4]. Contradiction detection algorithm:
```python
def detect_contradiction(program):
    for predicate in program.predicates():
        if prove(program, predicate) and prove(program, Not(predicate)):
            return True
    return False
```

Python library options for Datalog:
1. **pyDatalog** (https://github.com/pcarbonn/pyDatalog): Note that this package is unmaintained; the README recommends using IDP-Z3 instead [5]
2. **pyswip** (https://github.com/yuce/pyswip): Python interface to SWI-Prolog, suitable for Prolog-based Datalog evaluation
3. **Custom implementation**: For constrained Datalog fragment, a simple evaluator can be implemented

## Metropolis-Hastings MCMC for Discrete State Space

The MCMC algorithm generates samples from the energy function using Metropolis-Hastings with the acceptance criterion [6]:

acceptance_probability = min(1, exp(-(E_proposed - E_current) / T))

where T is the temperature parameter.

Algorithm:
```python
def metropolis_hastings(initial, energy_fn, proposal_fn, T, n_iter):
    current = initial
    current_energy = energy_fn(current)
    samples = [current]
    for i in range(n_iter):
        proposed = proposal_fn(current)
        proposed_energy = energy_fn(proposed)
        delta = proposed_energy - current_energy
        if random.random() < math.exp(-delta / T):
            current = proposed
            current_energy = proposed_energy
        samples.append(current)
    return samples
```

Proposal distributions for logic programs [6]:
- **Add**: Randomly select fragment from pool F, add to program (probability 0.5)
- **Remove**: Randomly select clause from program, remove it (probability 0.3)
- **Swap**: Remove one clause, add one fragment (probability 0.2)

Ensure proposed program remains valid Datalog (no function symbols, stratified negation).

Temperature schedule recommendations:
- Fixed T=1.0 for initial experiments
- Ablation: T ∈ {0.1, 0.5, 1.0, 2.0}
- Convergence check: Track energy over iterations, stop if stable for 100 steps

## Prolog/pyswip Integration

SWI-Prolog integration via pyswip provides syntax validation and query execution [7].

Installation:
```bash
# Install SWI-Prolog
sudo apt-get install swi-prolog  # Linux
# Install pyswip
pip install pyswip
```

Basic usage:
```python
from pyswip import Prolog
prolog = Prolog()
prolog.assertz('parent(john, mary)')
result = list(prolog.query('parent(john, X)'))
```

Syntax validation:
```python
def is_valid_prolog(code_string):
    try:
        prolog = Prolog()
        prolog.assertz(code_string)
        return True
    except Exception:
        return False
```

Note: prolog.assertz() actually asserts the fact; for pure syntax check, use consult with temp file.

## Unified Notation and Weights

The complete energy function is defined as:

E(P, T, Q) = w1*E_text(P, T) + w2*E_onto(P) + w3*E_cohere(P, Q)

Where:
- T: Input text
- Q: Query
- P: Program (set of Datalog clauses)
- Fragment pool F = {f_1, ..., f_n}, |F| ≤ 50
- Weights from hypothesis: w1=0.4, w2=0.3, w3=0.3

## Implementation Library Summary

| Component | Library | Version | Purpose |
|-----------|---------|---------|--------|
| E_text | sentence-transformers | 2.2.2+ | SBERT embeddings |
| E_onto | requests | 2.31.0+ | ConceptNet API calls |
| E_cohere | pyswip or pyDatalog | - | Datalog reasoning |
| MCMC | numpy, random | - | Sampling algorithm |
| Prolog | pyswip | 0.3.3+ | Syntax check, query |

## Failure Scenarios and Mitigations

1. **ConceptNet API down**: Use cached responses or download ConceptNet subset locally (SQLite)
2. **Prolog syntax check slow**: Pre-validate with regex before Prolog
3. **MCMC non-convergence**: Increase iterations, adjust temperature
4. **SBERT OOM**: Use smaller model (all-MiniLM-L6-v2: 80MB vs all-mpnet-base-v2: 420MB)

## Confidence Assessment

Confidence level: MEDIUM-HIGH
- Mathematical formulations for E_text and MCMC are well-established in literature [1, 6]
- ConceptNet API documentation is comprehensive but API availability was not confirmed during research [3]
- Datalog semantics are standard but library recommendations have caveats (pyDatalog unmaintained) [4, 5]
- Prolog integration via pyswip is well-documented but requires system-level SWI-Prolog installation [7]

Would change confidence to HIGH with:
- Successful live test of ConceptNet API
- Verification of pyDatalog/IDP-Z3 for contradiction detection
- Benchmark of MCMC convergence on sample programs

## Sources

[1] [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Original paper introducing SBERT with cosine similarity for semantic textual similarity

[2] [Sentence Transformers Pretrained Models Documentation](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) — Official documentation comparing all-MiniLM-L6-v2 (fast, 384-dim) vs all-mpnet-base-v2 (best quality, 768-dim)

[3] [ConceptNet API Documentation](https://github.com/commonsense/conceptnet5/wiki/API) — Documents REST API endpoints, query format, relation types, and usage patterns for ConceptNet knowledge graph

[4] [Datalog - Wikipedia](https://en.wikipedia.org/wiki/Datalog) — Covers Datalog syntax, semantics (model-theoretic, fixed-point, proof-theoretic), and stratified negation

[5] [pyDatalog GitHub Repository](https://github.com/pcarbonn/pyDatalog) — Python Datalog library - README warns package is unmaintained and recommends IDP-Z3 instead

[6] [Metropolis-Hastings Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) — Describes MCMC algorithm with acceptance ratio min(1, f(x')/f(x_t)) for discrete state spaces

[7] [pyswip GitHub Repository](https://github.com/yuce/pyswip) — Python interface to SWI-Prolog for logic programming and query execution

## Follow-up Questions

- What is the exact performance benchmark comparing all-MiniLM-L6-v2 vs all-mpnet-base-v2 on semantic textual similarity tasks relevant to logic program representations?
- How does the IDP-Z3 system (recommended by pyDatalog maintainers) compare to custom Datalog evaluation for contradiction detection in constrained logic fragments?
- What are the empirical convergence rates of Metropolis-Hastings with the proposed proposal distribution (add/remove/swap) on typical logic program synthesis tasks with |F| ≤ 50 fragments?

---
*Generated by AI Inventor Pipeline*
