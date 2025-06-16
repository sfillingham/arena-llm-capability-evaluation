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
    filter_dataset,
    generate_and_score_questions
)

from .inspect_evaluation import (
    record_to_sample,
    record_to_sample_full,
    prompt_template,
    letters_and_answer_options,
    multiple_choice_format,
    make_choice,
    self_critique_format,
    benchmark_eval,
    alignment_eval
)

from .statistical_analysis import (
    split_eval_samples,
    plot_question_bias_type_by_question_topic,
    plot_answer_type_by_question_topic
)

from .utils import (
    add_few_shot_examples,
    add_variance_prompts,
    GenPrompts,
    pretty_print_questions
)

# Main workflow functions
__all__ = [
    # Question generation
    "retry_with_exponential_backoff",
    "generate_structured_response",
    "generate_structured_responses_with_threadpool",
    "filter_dataset",
    "generate_and_score_questions",
    
    # Evaluation
    "record_to_sample",
    "record_to_sample_full",
    "prompt_template",
    "letters_and_answer_options",
    "multiple_choice_format",
    "make_choice",
    "self_critique_format",
    "benchmark_eval",
    "alignment_eval",
    
    # Analysis
    "split_eval_samples",
    "plot_question_bias_type_by_question_topic",
    "plot_answer_type_by_question_topic",
    
    # Utilities
    "add_few_shot_examples",
    "add_variance_prompts", 
    "GenPrompts",
    "pretty_print_questions"
]