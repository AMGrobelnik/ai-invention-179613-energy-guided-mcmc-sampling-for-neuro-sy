import json

# SBERT Model Information
sbert_models = {
    "all-MiniLM-L6-v2": {
        "description": "Fast, 5x faster than all-mpnet-base-v2, good quality",
        "embedding_dim": 384,
        "speed": "fast",
        "quality": "good"
    },
    "all-mpnet-base-v2": {
        "description": "Best quality, general purpose model",
        "embedding_dim": 768,
        "speed": "slower",
        "quality": "best"
    }
}

# Cosine Similarity Formula
cosine_formula = {
    "formula": "cos_sim(a, b) = (a · b) / (||a|| * ||b||)",
    "latex": "\\cos(a, b) = \\frac{a \\cdot b}{\\|a\\| \\|b\\|}",
    "e_text": "E_text = 1 - cos_sim(SBERT(text), SBERT(program))"
}

# ConceptNet API Information
conceptnet_info = {
    "base_url": "http://api.conceptnet.io",
    "endpoints": {
        "concept_lookup": "/c/en/{concept}",
        "query": "/query?start=/c/en/{concept1}&end=/c/en/{concept2}&rel=/r/{relation}"
    },
    "relation_types": [
        "/r/IsA", "/r/PartOf", "/r/HasA", "/r/UsedFor",
        "/r/CapableOf", "/r/AtLocation", "/r/Causes", "/r/HasSubevent",
        "/r/HasFirstSubevent", "/r/HasLastSubevent", "/r/HasPrerequisite",
        "/r/MotivatedByGoal", "/r/CausesDesire", "/r/Desires",
        "/r/DefinedAs", "/r/MeasuredBy", "/r/HasProperty", "/r/ReceivesAction"
    ],
    "rate_limits": "Not officially documented, be polite with ~1 request/second",
    "note": "API returned 502 error during research - may be down or rate-limited"
}

# Datalog Information
datalog_info = {
    "semantics": ["Model-theoretic", "Fixed-point", "Proof-theoretic"],
    "stratified_negation": "Negation only allowed in later strata to avoid inconsistency",
    "contradiction_detection": "If P and not P are both entailed, contradiction exists",
    "python_libs": {
        "pyDatalog": "https://github.com/pcarbonn/pyDatalog",
        "pyswip": "https://github.com/yuce/pyswip",
        "custom": "Simple Datalog evaluator can be implemented for constrained fragment"
    }
}

# Metropolis-Hastings Algorithm
mcmc_algorithm = {
    "type": "Metropolis-Hastings for discrete state space",
    "acceptance_criterion": "min(1, exp(-delta_energy / temperature))",
    "proposal_types": ["add", "remove", "swap"],
    "proposal_details": {
        "add": "Randomly select fragment from pool, add to program",
        "remove": "Randomly select clause from program, remove it",
        "swap": "Remove one clause, add one fragment"
    }
}

# Prolog/pyswip Information
prolog_info = {
    "pyswip_install": "pip install pyswip",
    "swi_prolog_install": "sudo apt-get install swi-prolog (Linux)",
    "basic_usage": "from pyswip import Prolog; prolog = Prolog()",
    "syntax_check": "Use try-except with prolog.assertz()"
}

# Save all information
research_data = {
    "sbert_models": sbert_models,
    "cosine_formula": cosine_formula,
    "conceptnet": conceptnet_info,
    "datalog": datalog_info,
    "mcmc": mcmc_algorithm,
    "prolog": prolog_info
}

with open("research_interim.json", "w") as f:
    json.dump(research_data, f, indent=2)

print("Research data saved to research_interim.json")
print(json.dumps(research_data, indent=2))
