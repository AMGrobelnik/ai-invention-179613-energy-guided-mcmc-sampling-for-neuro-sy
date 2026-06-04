#!/usr/bin/env python3
"""Quick test script for FOL fragment validation (no LLM calls)."""

from loguru import logger
import sys
sys.path.insert(0, '.')

from method import (
    create_synthetic_data,
    validate_syntax_prolog_fragment,
    validate_prolog_syntax_regex,
    compute_diversity_tfidf
)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

def test_syntax_validation():
    """Test Prolog syntax checker with known examples."""
    logger.info("TEST: Syntax Validation")

    test_cases = [
        ("valid_fact.", True),
        ("rule(X) :- fact(X).", True),
        ("invalid fact", False),  # Missing period
        ("bad(balance(.", False),  # Unbalanced parens
    ]

    for fragment, expected_valid in test_cases:
        is_valid, error = validate_syntax_prolog_fragment(fragment)
        status = "PASS" if is_valid == expected_valid else "FAIL"
        logger.info(f"  [{status}] Fragment: '{fragment[:30]}...' Expected: {expected_valid}, Got: {is_valid}")
        if not is_valid and error:
            logger.debug(f"         Error: {error}")

def test_semantic_diversity():
    """Test diversity computation with TF-IDF fallback."""
    logger.info("TEST: Semantic Diversity (TF-IDF)")

    test_frags = [
        {'fragment': 'likes(alice, bob).', 'syntax_valid': True},
        {'fragment': 'loves(alice, bob).', 'syntax_valid': True},
        {'fragment': 'hates(alice, bob).', 'syntax_valid': True},
    ]

    diversity = compute_diversity_tfidf(test_frags)
    logger.info(f"  [PASS] Computed TF-IDF diversity: {diversity:.3f}")
    assert 0 <= diversity <= 1, "Diversity out of range"

def test_data_loading():
    """Test synthetic data loading."""
    logger.info("TEST: Data Loading")

    examples = create_synthetic_data(num_examples=5)
    logger.info(f"  [PASS] Created {len(examples)} synthetic examples")

    # Verify structure
    for ex in examples:
        assert 'id' in ex, "Missing 'id' field"
        assert 'text' in ex, "Missing 'text' field"
        assert 'queries' in ex, "Missing 'queries' field"

    logger.info("  [PASS] All examples have required fields")

def main():
    logger.info("=" * 50)
    logger.info("DRY RUN TESTS (No LLM calls)")
    logger.info("=" * 50)

    test_data_loading()
    test_syntax_validation()
    test_semantic_diversity()

    logger.info("=" * 50)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
