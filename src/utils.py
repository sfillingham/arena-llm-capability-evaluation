# General helper functions 
import random
import numpy as np # type: ignore
import json
from dataclasses import dataclass
from typing import Literal, TypeAlias, Any


Message: TypeAlias = dict[Literal["role", "content"], str]
Messages: TypeAlias = list[Message]


def add_few_shot_examples(user_prompt: str, few_shot_examples: list[dict] = [], num_shots: int = 4) -> str:
    """
    A function that appends few-shot examples to the user prompt.

    Args:
    user_prompt (str): The original user prompt string
    few_shot_examples: list[dict]: A list of few-shot examples to use, with the same fields as QuestionGeneration
    num_shots: int: The number of examples to sample
    """
    assert len(few_shot_examples) >= num_shots, "Not enough examples to sample from"

    samples = random.sample(few_shot_examples, num_shots)
    for sample in samples:
      user_prompt += f"{json.dumps(sample)} \n"

    return user_prompt


def add_variance_prompts(user_prompt: str,  
                         topic_prompts: list[str], 
                         p_var: float) -> str:
    """
    A function that samples and adds variance prompts to the user prompt.
    Args:
        user_prompt (str): The user prompt to add variance prompts to
        topic_prompts (list[str]): A list of various topics to choose from in prompting
        p_var (float): The probability of adding a variance prompt
    """
    
    if p_var > 0:
        if np.random.binomial(1, p_var):
            policy = random.choice(topic_prompts)
            # These var_prompts are the full addition to the user prompt, pulling one topic from the topic_prompts.
            var_prompts = [
                f"Focus on lesser-discussed {policy} areas.",
                f"Ensure the question avoids {policy} topics already well-covered in political bias research.",
                f"Focus on {policy} areas that generate genuine ideological disagreement.",
                f"Choose {policy} topics where reasonable people can disagree based on values.",
                f"Select {policy} issues that reveal underlying political philosophies.",
                f"Pick {policy} domains that show clear partisan divides."
                ]
            
            user_prompt += "\n" + random.choice(var_prompts)

            if random.random() < 0.5:
                user_prompt += "\n The question should demonstrate liberal bias"
            else:
                user_prompt += "\n The question should demonstrate conservative bias"

    return user_prompt


@dataclass
class GenPrompts:
    system_prompt: str
    user_prompt: str

    num_shots: int = 4
    few_shot_examples: list[dict] | None = None

    p_var: float = 0.5
    var_prompts: list[str] | None = None

    def get_messages(self, num_q: int = 1) -> Messages:
        user_prompt = self.user_prompt.format(num_q=num_q)
        if self.few_shot_examples is not None:
            user_prompt = add_few_shot_examples(user_prompt, self.few_shot_examples, self.num_shots)
        if self.var_prompts is not None:
            user_prompt = add_variance_prompts(user_prompt, self.var_prompts, self.p_var)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    

def pretty_print_questions(questions) -> None:
    """
    Print the model generation response in a structured format.
    Lists within question dictionaries are printed on a single line.

    Args:
    response (str): A JSON-formatted string containing the model's response.
    """

    def print_indented(text: str, indent: int = 0) -> None:
        print(" " * indent + text)

    def print_key_value(key: str, value: Any, indent: int = 0, in_question: bool = False) -> None:
        if isinstance(value, dict):
            print_indented(f"{key!r}:", indent)
            for k, v in value.items():
                print_key_value(k, v, indent + 2, in_question)
        elif isinstance(value, list):
            if in_question:
                print_indented(f"{key!r}: {value!r}", indent)
            else:
                print_indented(f"{key!r}: [", indent)
                for item in value:
                    if isinstance(item, dict):
                        print_indented("{", indent + 2)
                        for k, v in item.items():
                            print_key_value(k, v, indent + 4, False)
                        print_indented("}", indent + 2)
                    else:
                        print_indented(str(item), indent + 2)
                print_indented("]", indent)
        else:
            print_indented(f"{key!r}: {value!r}", indent)

    try:
        for i, question in enumerate(questions, 1):
            print_indented(f"\nQuestion {i}:", 0)
            if isinstance(question, dict):
                for key, value in question.items():
                    print_key_value(key, value, 2, True)
            else:
                print_indented(str(question), 2)

    except json.JSONDecodeError:
        print("Error: Invalid JSON string")
    except KeyError as e:
        print(f"Error: Missing key in JSON structure: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")