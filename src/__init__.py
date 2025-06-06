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
    generate_political_questions,
    validate_question_format,
    save_question_set
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
    load_json,
    save_results,
    format_results_table
)

# Main workflow functions
__all__ = [
    # Question generation
    "generate_political_questions",
    "validate_question_format", 
    "save_question_set",
    
    # Evaluation
    "run_inspect_evaluation",
    "setup_evaluation_config",
    "EvaluationRunner",
    
    # Analysis
    "analyze_bias_patterns",
    "calculate_bias_metrics",
    "generate_comparison_report",
    
    # Utilities
    "load_json",
    "save_results", 
    "format_results_table"
]