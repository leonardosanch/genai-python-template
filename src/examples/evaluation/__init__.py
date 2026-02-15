"""Evaluation examples for GenAI systems."""

from src.examples.evaluation.deepeval_hallucination import (
    evaluate_faithfulness,
    evaluate_hallucination_batch,
)
from src.examples.evaluation.llm_as_judge import (
    pairwise_comparison,
    rubric_based_evaluation,
)
from src.examples.evaluation.ragas_evaluation import (
    compare_chunking_strategies,
    evaluate_rag_pipeline,
)
from src.examples.evaluation.regression_testing import (
    detect_output_drift,
    run_regression_tests,
)

__all__ = [
    "evaluate_faithfulness",
    "evaluate_hallucination_batch",
    "evaluate_rag_pipeline",
    "compare_chunking_strategies",
    "pairwise_comparison",
    "rubric_based_evaluation",
    "run_regression_tests",
    "detect_output_drift",
]
