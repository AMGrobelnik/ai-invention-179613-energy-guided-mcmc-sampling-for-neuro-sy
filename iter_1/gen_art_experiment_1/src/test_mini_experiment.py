#!/usr/bin/env python3
"""Mini experiment test with mock LLM (no API costs)."""

from loguru import logger
import sys
import json
import numpy as np
from pathlib import Path
sys.path.insert(0, '.')

from method import (
    BudgetedLLMCaller,
    create_synthetic_data,
    validate_syntax_prolog_fragment,
    process_example,
)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

# Mock the LLM call to avoid API costs
def mock_call_with_budget(self, model, prompt, temperature, max_tokens=2000):
    """Mock LLM response for testing."""
    mock_responses = [
        f"likes(alice, bob).\nloves(bob, cat).\nfriend(X, Y) :- likes(X, Y), likes(Y, X).",
        f"loves(alice, bob).\nfriend(bob, alice).\nenemy(X, Y) :- hates(X, Y).",
        f"happy(alice).\nsad(bob).\nrich(X) :- has_money(X), not poor(X).",
    ]

    import random
    response = random.choice(mock_responses)
    cost = 0.001  # Mock cost
    self.total_cost += cost
    self.call_count += 1
    return response, cost

# Monkey-patch for testing
BudgetedLLMCaller.call_with_budget = mock_call_with_budget

def test_mini_experiment():
    """Test mini experiment with 1 example."""
    logger.info("=" * 50)
    logger.info("MINI EXPERIMENT (1 example, mock LLM)")
    logger.info("=" * 50)

    # Create test example
    examples = create_synthetic_data(num_examples=1)

    # Initialize budget caller
    budget_caller = BudgetedLLMCaller(max_budget_usd=0.50)

    # Process example (without Sentence-BERT to avoid GPU requirement)
    logger.info("Processing example with mock LLM...")
    result = process_example(
        examples[0],
        budget_caller,
        sbert_model=None,
        use_tfidf_fallback=True
    )

    # Print results
    logger.info(f"Example {result['example_id']} processed")
    logger.info(f"  Fragments: {len(result['fragments'])}")
    logger.info(f"  Valid rate: {result['valid_fragment_rate']:.0%}")
    logger.info(f"  Diversity: {result['semantic_diversity_mean']:.3f}")
    logger.info(f"  Cost: ${budget_caller.total_cost:.4f}")

    # Validate output format
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj

    output = convert_numpy_types({
        'experiment_name': 'pilot_fol_fragment_validation',
        'metrics': {
            'total_examples': 1,
            'total_fragments_generated': len(result['fragments']),
            'total_valid_fragments': result['valid_fragment_count'],
            'overall_valid_rate': result['valid_fragment_rate'],
        },
        'results_per_example': [result],
    })

    output_path = Path("test_mini_output.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved test output to {output_path}")

    logger.info("=" * 50)
    logger.info("MINI EXPERIMENT COMPLETED")
    logger.info("=" * 50)

    return result

if __name__ == "__main__":
    test_mini_experiment()
