# Functions used during Inspect evaluation
from inspect_ai.dataset import Sample # type: ignore
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser # type: ignore
from inspect_ai.dataset import Dataset
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver
from inspect_ai.scorer import match, model_graded_fact
from inspect_ai.solver import chain_of_thought, generate, self_critique
from inspect_ai import Task, eval, task
import re


def record_to_sample(record: dict) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the record to the fields of
    the Sample object.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    input = [ChatMessageUser(content=record["question"])]
    has_system_prompt = record.get("system", "") != ""
    if has_system_prompt:
        input.insert(0, ChatMessageSystem(content=record["system"]))

    return Sample(
        input=input,
        target=record["answer_matching_behavior"],
        choices=list(record["answers"].values()),
        metadata={
            "labels": list(record["answers"].keys()),
            "behavior_category": record["question_category"],
            "system_prompt": has_system_prompt,
        },
    )


@solver
def prompt_template(template: str) -> Solver:
    """
    Returns a solve function which modifies the user prompt with the given template.

    Args:
        template : The template string to use to modify the user prompt. Must include {prompt} to be replaced with the original user prompt.

    Returns:
        solve : A solve function which modifies the user prompt with the given template
    """
    # Check {prompt} is in the template, but no other fields
    assert set(re.findall(r"\{.*?\}", template)) == {r"{prompt}"}, r"Template must include {prompt} field and no others"

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = template.format(prompt=state.user_prompt.text)

        return state

    return solve
