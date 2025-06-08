"""
ARENA LLM Capability Evaluation

A framework for evaluating political bias in frontier AI models using the Inspect framework.
Developed as part of ARENA Chapter 3 curriculum with extensions for systematic capability assessment.

This package provides tools for:
- Generating evaluation questions targeting specific threat models
- Running systematic evaluations using the Inspect framework  
- Statistical analysis of model responses and bias detection
"""

__version__ = "0.1.0"
__author__ = "Sean P. Fillingham"

# Core functionality imports
from .question_generation import (
    retry_with_exponential_backoff,
    generate_structured_response,
    generate_structured_responses_with_threadpool,
    filter_dataset
)

from .inspect_evaluation import (
    run_inspect_evaluation,
    setup_evaluation_config,
    EvaluationRunner
)

from .statistical_analysis import (
    analyze_bias_patterns,
    calculate_bias_metrics,
    generate_comparison_report
)

from .utils import (
    add_few_shot_examples,
    add_variance_prompts,
    GenPrompts
)

# Main workflow functions
__all__ = [
    # Question generation
    "retry_with_exponential_backoff",
    "generate_structured_response",
    "generate_structured_responses_with_threadpool",
    "filter_dataset",
    
    # Evaluation
    "run_inspect_evaluation",
    "setup_evaluation_config",
    "EvaluationRunner",
    
    # Analysis
    "analyze_bias_patterns",
    "calculate_bias_metrics",
    "generate_comparison_report",
    
    # Utilities
    "add_few_shot_examples",
    "add_variance_prompts", 
    "GenPrompts"
]