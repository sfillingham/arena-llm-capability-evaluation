# arena-llm-capability-evaluation
Development of AI capability evaluations using [Inspect](https://inspect.aisi.org.uk/) from AISI

In this currently ongoing project, I generally follow the flow of [Chapter 3](https://arena3-chapter3-llm-evals.streamlit.app/) in the [ARENA](https://www.arena.education/) curriculum, however I have refactored and expanded their code to serve my purposes in this project.

## Project Description
For this evaluation I develop a political bias threat model to assess how frontier AI models might exhibit systematic political biases that could influence democratic processes or policy discussions. I create a set of 300 multiple-choice questions and implement them using the Inspect framework for rigorous model evaluation.

As a secondary component, I utilize an older (Claude-3.7) model to assist in developing evaluation questions for current frontier models. This approach demonstrates a critical AI safety methodology: using currently trusted models to evaluate the safety and alignment of more capable systems.

## Methodology
- Threat Model: Political bias in AI responses that could systematically favor particular political perspectives
- Question Design: 300 MCQs covering policy, socioeconomic, and political reasoning.
- Evaluation Framework: Inspect-based automated assessment with statistical analysis
- Model Comparison: Testing across multiple frontier models (GPT-4.1, Claude-4, etc.)

## How to run the code
### Environment Setup
The required packages are in the requirements.txt file.
To install run the following
```
pip install -r requirements.txt
```

## Current Results and Ongoing Work
Models Evaluated:
- GPT-4o-mini
- GPT-4.1-mini
- Claude Sonnet 3.7
- Claude Sonnet 4

Initial results:
- Developed political bias threat model and evaluation.
- Current results are being analyzed, however a couple of interesting, VERY preliminary results:
    - All models show little to no significant political bias
    - Whenever a model does chose a biased answer, it was liberal leaning, suggesting a low-level, liberal bias in certain models for certain question topics.
    - All models do display varying levels of evasive behavior which will be further explored and likely demonstrates post-training security features.

Ongoing Work:
- Finish preliminary statistical analysis.
- Vary the prompt methods to attempt to elicit political bias and bypass the safety training.
- Further test on other models: Claude Opus 4, Grok 3, etc.

