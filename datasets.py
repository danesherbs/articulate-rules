# %%

import os
import random
import json
import utils
import transformers
import itertools
import functools
import pprint

from typing import Callable, Sequence, Optional, Tuple, List, Set, Literal
from dataclasses import dataclass
from typeguard import typechecked


# %%

DEFAULT_EXAMPLE_SEPERATOR = "\n\n"
DEFAULT_TASK_SEPERATOR = "\n\n###\n\n"
DEFAULT_PROMPT_INSTRUCTION = (
    "The following inputs are labelled 'True' if they match a pattern and 'False'"
    " otherwise. The pattern is known to be very simple and explainable in plain"
    " English. Label the remaining inputs according to the pattern."
)
DEFAULT_MULTIPLE_CHOICE_QUESTION = (
    "Question: What is the most likely pattern being used to label the inputs above?"
)
DEFAULT_BINARY_CLASSIFICATION_EXAMPLE_TO_PROBLEM_FN = lambda example: Problem(
    prompt=f"Input: {example.value}\nLabel:",
    expected_completion=f" {example.label}",
)

# %%


@dataclass(frozen=True)
class BinaryClassificationExample:
    value: str
    label: bool

    def __repr__(self, indent=0):  # quick hack; remove this or tidy up
        indents = "\t" * indent
        return f'{indents}BinaryClassificationExample(\n{indents}\tvalue="{self.value}",\n{indents}\tlabel={self.label}\n{indents})'


@dataclass(frozen=True)
class BinaryClassificationTask:
    instruction: Optional[str]
    examples: Sequence[BinaryClassificationExample]
    chain_of_thought: Optional[str] = None

    def __repr__(self, indent=0):  # quick hack; remove this or tidy up
        indents = "\t" * indent
        examples_repr = ",\n".join(
            [example.__repr__(indent + 2) for example in self.examples]
        )
        return (
            "BinaryClassificationTask(\n"
            f"{indents}\texamples=(\n{examples_repr}\n{indents}\t),\n"
            f'{indents}\tinstruction="{self.instruction}",\n'
            f'{indents}\tchain_of_thought="{self.chain_of_thought}"\n{indents})'
        )


@dataclass(frozen=True)
class FreeformExplanationTask:
    instruction: Optional[str]
    classification_task: BinaryClassificationTask
    explanation: str
    chain_of_thought: Optional[str] = None

    def __repr__(self, indent=0):  # quick hack; remove this or tidy up
        indents = "\t" * indent
        return (
            "FreeformExplanationTask(\n"
            f'{indents}\tinstruction="{self.instruction}",\n'
            f"{indents}\tclassification_task={self.classification_task.__repr__(indent+1)},\n"
            f'{indents}\texplanation="{self.explanation}",\n'
            f'{indents}\tchain_of_thought="{self.chain_of_thought}"\n{indents})'
        )

    def set_chain_of_thought(self, chain_of_thought: str) -> "FreeformExplanationTask":
        return FreeformExplanationTask(
            instruction=self.instruction,
            classification_task=self.classification_task,
            explanation=self.explanation,
            chain_of_thought=chain_of_thought,
        )


@dataclass(frozen=True)
class MultipleChoiceExplanationTask:
    instruction: Optional[str]
    classification_task: BinaryClassificationTask
    choices: Sequence[str]
    answer: str


@dataclass(frozen=True, order=True)
class Problem:
    prompt: str
    expected_completion: str


# %%


class BinaryClassificationExampleGenerationBehaviour:
    def generate_passing_example(self) -> BinaryClassificationExample:
        """Generate an example that matches the pattern."""
        raise NotImplementedError()

    def generate_failing_example(self) -> BinaryClassificationExample:
        """Generate an example that does not match the pattern."""
        raise NotImplementedError()


# %%


class BinaryClassificationDiscriminationBehaviour:
    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        """Returns whether the example matches the pattern."""
        raise NotImplementedError()


# %%


@typechecked
class Rule:
    def __init__(
        self,
        explanation: str,
        generation_behaviour: BinaryClassificationExampleGenerationBehaviour,
        discrimination_behaviour: BinaryClassificationDiscriminationBehaviour,
    ):
        self.explanation = explanation
        self.generation_behaviour = generation_behaviour
        self.discrimination_behaviour = discrimination_behaviour

        assert len(explanation) > 0, "Explanation must be non-empty"
        assert (
            self.generation_behaviour.generate_passing_example
            != self.generation_behaviour.generate_failing_example
        ), "Passing and failing example generation functions must be different"
        assert discrimination_behaviour.is_passing_example(
            generation_behaviour.generate_passing_example()
        ), "Discriminator must label passing examples as passing"
        assert not discrimination_behaviour.is_passing_example(
            generation_behaviour.generate_failing_example()
        ), "Discriminator must label failing examples as failing"

    def generate_passing_example(self) -> BinaryClassificationExample:
        return self.generation_behaviour.generate_passing_example()

    def generate_failing_example(self) -> BinaryClassificationExample:
        return self.generation_behaviour.generate_failing_example()

    def generate_binary_classification_task(
        self,
        n_passing_examples,
        n_failing_examples,
        instruction: Optional[str] = DEFAULT_PROMPT_INSTRUCTION,
        ends_with: Literal["passing"] | Literal["failing"] | None = None,
    ) -> BinaryClassificationTask:
        """Generates a `BinaryClassificationTask` for the rule."""

        return generate_binary_classification_task(
            instruction=instruction,
            generate_passing_example=self.generation_behaviour.generate_passing_example,
            generate_failing_example=self.generation_behaviour.generate_failing_example,
            n_passing_examples=n_passing_examples,
            n_failing_examples=n_failing_examples,
            ends_with=ends_with,
        )

    def generate_freeform_explanation_task(
        self,
        n_passing_examples: int,
        n_failing_examples: int,
        instruction: Optional[str] = DEFAULT_PROMPT_INSTRUCTION,
    ) -> FreeformExplanationTask:
        """Generates a `FreeformExplanationTask` task for the rule."""

        assert (
            n_passing_examples >= 0
        ), "Number of passing examples must be non-negative"
        assert (
            n_failing_examples >= 0
        ), "Number of passing examples must be non-negative"
        assert (
            n_passing_examples + n_failing_examples > 0
        ), "Must have at least one example in the task"

        classification_task = self.generate_binary_classification_task(
            instruction=None,
            n_passing_examples=n_passing_examples,
            n_failing_examples=n_failing_examples,
        )

        freeform_explanation_task = FreeformExplanationTask(
            instruction=instruction,
            classification_task=classification_task,
            explanation=self.explanation,
        )

        return freeform_explanation_task

    def generate_multiple_choice_explanation_task(
        self,
        n_passing_examples: int,
        n_failing_examples: int,
        incorrect_choices: Sequence[str],
        instruction: Optional[str] = DEFAULT_PROMPT_INSTRUCTION,
    ) -> MultipleChoiceExplanationTask:
        """Generates a `MultipleChoiceExplanationTask` task for the rule."""

        assert (
            n_passing_examples >= 0
        ), "Number of passing examples must be non-negative"
        assert (
            n_failing_examples >= 0
        ), "Number of passing examples must be non-negative"
        assert (
            n_passing_examples + n_failing_examples > 0
        ), "Must have at least one example in the task"
        assert (
            self.explanation not in incorrect_choices
        ), "`incorrect_choices` must contain incorrect answers only"
        assert len(incorrect_choices) >= 1, "Must provide at least one inccorect choice"
        assert len(set(incorrect_choices)) == len(
            incorrect_choices
        ), "Choices must be unique"

        classification_task = self.generate_binary_classification_task(
            instruction=None,
            n_passing_examples=n_passing_examples,
            n_failing_examples=n_failing_examples,
        )

        n_choices = len(incorrect_choices) + 1
        choices = random.sample(
            (self.explanation,) + tuple(incorrect_choices), k=n_choices
        )

        assert (
            self.explanation in choices
        ), f"Correct explanation `{self.explanation}` must be in choices"
        assert (
            len(choices) == n_choices
        ), f"Expected to have {n_choices} choices, but got {len(choices)}"
        assert (
            len(set(choices)) == n_choices
        ), f"Expected to have {n_choices} unique choices, but got {len(set(choices))}"

        return MultipleChoiceExplanationTask(
            instruction=instruction,
            classification_task=classification_task,
            choices=choices,
            answer=self.explanation,
        )

    def __call__(self, example: BinaryClassificationExample) -> bool:
        return self.discrimination_behaviour.is_passing_example(example)


# %%


def generate_binary_classification_task(
    instruction: Optional[str],
    generate_passing_example: Callable[[], BinaryClassificationExample],
    generate_failing_example: Callable[[], BinaryClassificationExample],
    n_passing_examples,
    n_failing_examples,
    ends_with: Literal["passing"] | Literal["failing"] | None = None,
) -> BinaryClassificationTask:
    """Generates a `BinaryClassificationTask`."""

    # pre-conditions
    assert n_passing_examples >= 0, "Number of passing examples must be non-negative"
    assert n_failing_examples >= 0, "Number of failing examples must be non-negative"
    assert n_passing_examples + n_failing_examples > 0, "Must have at least one example"
    assert (
        generate_passing_example != generate_failing_example
    ), "Passing and failing example functions must be different"

    if ends_with == "passing":
        assert (
            n_passing_examples >= 1
        ), "Must have at least one passing example if `ends_with` is `passing`"

    if ends_with == "failing":
        assert (
            n_failing_examples >= 1
        ), "Must have at least one failing example if `ends_with` is `failing`"

    # generate examples
    pass_examples = [generate_passing_example() for _ in range(n_passing_examples)]
    fail_examples = [generate_failing_example() for _ in range(n_failing_examples)]
    examples = pass_examples + fail_examples
    examples = tuple(random.sample(examples, k=len(examples)))

    if ends_with == "passing":
        random_pass_example = random.choice(pass_examples)
        examples = utils.sample(
            examples, k=len(examples) - 1, exclude={random_pass_example}
        ) + (random_pass_example,)

    if ends_with == "failing":
        random_fail_example = random.choice(fail_examples)
        examples = utils.sample(
            examples, k=len(examples) - 1, exclude={random_fail_example}
        ) + (random_fail_example,)

    # post-conditions
    assert all(
        isinstance(example, BinaryClassificationExample) for example in examples
    ), "All generated examples must be of type `BinaryClassificationExample`"
    assert all(
        passing_example.label for passing_example in pass_examples
    ), "All passing examples must have label `True`"
    assert all(
        not failing_example.label for failing_example in fail_examples
    ), "All failing examples must have label `False`"
    assert (
        len(pass_examples) == n_passing_examples
    ), "`n_passing_examples` must be equal to number of generated passing examples"
    assert (
        len(fail_examples) == n_failing_examples
    ), "`n_failing_examples` must be equal to number of generated failing examples"
    assert (
        len(examples) == n_passing_examples + n_failing_examples
    ), "Number of examples must be equal to sum of passing and failing examples"

    return BinaryClassificationTask(
        instruction=instruction,
        examples=examples,
    )


# %%


def binary_classification_task_to_problem(
    task: BinaryClassificationTask,
    few_shot_tasks: Optional[Tuple[BinaryClassificationTask]] = None,
    example_to_problem_fn: Callable[
        [BinaryClassificationExample], Problem
    ] = DEFAULT_BINARY_CLASSIFICATION_EXAMPLE_TO_PROBLEM_FN,
    example_seperator: str = DEFAULT_EXAMPLE_SEPERATOR,
    task_seperator: str = DEFAULT_TASK_SEPERATOR,
) -> Problem:
    """Converts a `BinaryClassificationTask` task to a `Problem`."""

    assert isinstance(example_to_problem_fn, Callable), (
        "Example to problem function must be a callable, got"
        f" {example_to_problem_fn} instead."
    )
    assert len(task.examples) >= 0, "Task must have non-negative number of examples"

    prompt = ""

    if few_shot_tasks is not None:
        for few_shot_task in few_shot_tasks:
            few_shot_problem = binary_classification_task_to_problem(few_shot_task)
            prompt += few_shot_problem.prompt + few_shot_problem.expected_completion
            prompt += task_seperator

    if task.instruction is not None:
        prompt += task.instruction + "\n\n"

    for i, example in enumerate(task.examples):
        problem = example_to_problem_fn(example)
        prompt += problem.prompt

        if i == len(task.examples) - 1:
            break

        prompt += problem.expected_completion + example_seperator

    return Problem(
        prompt=prompt,
        expected_completion=problem.expected_completion,
    )


# %%


def binary_classification_task_with_cot_to_problem(
    task: BinaryClassificationTask,
    few_shot_tasks: Optional[Tuple[BinaryClassificationTask]] = None,
    example_to_problem_fn: Callable[
        [BinaryClassificationExample], Problem
    ] = DEFAULT_BINARY_CLASSIFICATION_EXAMPLE_TO_PROBLEM_FN,
    example_seperator: str = DEFAULT_EXAMPLE_SEPERATOR,
    task_seperator: str = DEFAULT_TASK_SEPERATOR,
) -> Problem:
    """Converts a `BinaryClassificationTask` task to a `Problem` with a chain of thought."""

    # pre-conditions
    assert isinstance(example_to_problem_fn, Callable), (
        "Example to problem function must be a callable, got"
        f" {example_to_problem_fn} instead."
    )
    assert len(task.examples) > 0, "Task must have non-negative number of examples"
    assert task.chain_of_thought is not None, "Task must have a chain of thought"

    # body
    prompt = ""

    if few_shot_tasks is None:
        few_shot_tasks = tuple()

    for few_shot_task in few_shot_tasks:
        few_shot_problem = binary_classification_task_with_cot_to_problem(few_shot_task)
        prompt += few_shot_problem.prompt
        prompt += few_shot_problem.expected_completion
        prompt += task_seperator

    if task.instruction is not None:
        prompt += task.instruction + "\n\n"

    for i, example in enumerate(task.examples):
        problem = example_to_problem_fn(example)

        if i == len(task.examples) - 1:
            break

        prompt += problem.prompt
        prompt += problem.expected_completion
        prompt += example_seperator

    prompt_with_explanation = problem.prompt.replace("Label:", f"Explanation:")  # type: ignore
    prompt += prompt_with_explanation

    expected_completion = (
        " " + task.chain_of_thought + "\nLabel:" + problem.expected_completion  # type: ignore
    )

    problem = Problem(
        prompt=prompt,
        expected_completion=expected_completion,
    )

    # post-conditions
    assert isinstance(problem, Problem), "Problem must be of type `Problem`"
    assert isinstance(problem.prompt, str), "Problem prompt must be of type `str`"
    assert isinstance(
        problem.expected_completion, str
    ), "Problem expected completion must be of type `str`"
    assert problem.prompt != "", "Problem prompt must not be empty"
    assert (
        problem.expected_completion != ""
    ), "Problem expected completion must not be empty"

    return problem


# %%


def freeform_explanation_task_to_problem(
    task: FreeformExplanationTask,
    example_to_problem_fn: Callable[
        [BinaryClassificationExample], Problem
    ] = DEFAULT_BINARY_CLASSIFICATION_EXAMPLE_TO_PROBLEM_FN,
    few_shot_tasks: Optional[Sequence[FreeformExplanationTask]] = None,
    example_seperator: str = DEFAULT_EXAMPLE_SEPERATOR,
    task_seperator: str = DEFAULT_TASK_SEPERATOR,
) -> Problem:
    """Converts a `FreeformExplanationTask` task to a `Problem`."""

    prompt = ""

    if task.instruction is not None:
        prompt += task.instruction + "\n\n"

    if few_shot_tasks is not None:
        for few_shot_task in few_shot_tasks:
            problem = binary_classification_task_to_problem(
                task=few_shot_task.classification_task,
                example_to_problem_fn=example_to_problem_fn,
                example_seperator=example_seperator,
            )

            prompt += problem.prompt + problem.expected_completion

            prompt += example_seperator
            prompt += f"Pattern: {few_shot_task.explanation}"
            prompt += task_seperator

    problem = binary_classification_task_to_problem(
        task=task.classification_task,
        example_to_problem_fn=example_to_problem_fn,
        example_seperator=example_seperator,
    )

    prompt += problem.prompt + problem.expected_completion
    prompt += example_seperator
    prompt += (
        "Question: What is the most likely pattern being used to label the inputs"
        " above?\n"
    )
    prompt += "Answer:"

    return Problem(
        prompt=prompt,
        expected_completion=f" {task.explanation}",
    )


# %%


def freeform_explanation_task_with_cot_to_problem(
    task: FreeformExplanationTask,
    example_to_problem_fn: Callable[
        [BinaryClassificationExample], Problem
    ] = DEFAULT_BINARY_CLASSIFICATION_EXAMPLE_TO_PROBLEM_FN,
    few_shot_tasks: Optional[Sequence[FreeformExplanationTask]] = None,
    example_seperator: str = DEFAULT_EXAMPLE_SEPERATOR,
    task_seperator: str = DEFAULT_TASK_SEPERATOR,
) -> Problem:
    """Converts a `FreeformExplanationTask` task to a `Problem` with a chain of thought."""

    assert task.chain_of_thought is not None, f"Task must have a chain of thought"

    prompt = ""

    if task.instruction is not None:
        prompt += task.instruction + "\n\n"

    if few_shot_tasks is not None:
        for few_shot_task in few_shot_tasks:
            assert (
                few_shot_task.chain_of_thought is not None
            ), f"Few-shot task must have a chain of thought"

            problem = binary_classification_task_to_problem(
                task=few_shot_task.classification_task,
                example_to_problem_fn=example_to_problem_fn,
                example_seperator=example_seperator,
            )

            prompt += problem.prompt + problem.expected_completion

            prompt += example_seperator
            prompt += (
                "Question: What is the most likely pattern being used to label the"
                " inputs above?\n"
            )
            prompt += f"Reasoning: {few_shot_task.chain_of_thought}\n"
            prompt += f"Answer: {few_shot_task.explanation}"
            prompt += task_seperator

    problem = binary_classification_task_to_problem(
        task=task.classification_task,
        example_to_problem_fn=example_to_problem_fn,
        example_seperator=example_seperator,
    )

    prompt += problem.prompt + problem.expected_completion

    prompt += example_seperator
    prompt += (
        "Question: What is the most likely pattern being used to label the inputs"
        " above?\n"
    )
    prompt += f"Reasoning:"

    expected_completion = f" {task.chain_of_thought}\Answer: {task.explanation}"

    return Problem(
        prompt=prompt,
        expected_completion=expected_completion,
    )


# %%


def multiple_choice_explanation_task_to_problem(
    task: MultipleChoiceExplanationTask,
    few_shot_tasks: Optional[Sequence[FreeformExplanationTask]] = None,
    question: str = DEFAULT_MULTIPLE_CHOICE_QUESTION,
    example_to_problem_fn: Callable[
        [BinaryClassificationExample], Problem
    ] = DEFAULT_BINARY_CLASSIFICATION_EXAMPLE_TO_PROBLEM_FN,
    example_seperator: str = DEFAULT_EXAMPLE_SEPERATOR,
    task_seperator: str = DEFAULT_TASK_SEPERATOR,
) -> Problem:
    """Converts a `MultipleChoiceExplanationTask` task to a `Problem`."""

    assert task.answer in task.choices, "Answer must appear in list of choices"
    assert len(task.choices) >= 2, "Must have at least two choices"
    assert (
        len(task.classification_task.examples) > 0
    ), "Must have at least one example in classification task"

    prompt = ""

    if task.instruction is not None:
        prompt += task.instruction + "\n\n"

    if few_shot_tasks is not None:
        for few_shot_task in few_shot_tasks:
            classification_problem = binary_classification_task_to_problem(
                task=few_shot_task.classification_task,
                example_to_problem_fn=example_to_problem_fn,
                example_seperator=example_seperator,
            )

            prompt += (
                classification_problem.prompt
                + classification_problem.expected_completion
            )

            prompt += example_seperator
            prompt += question + "\n"
            prompt += "Choices:" + "\n"

            multichoice_answer = None

            for i, choice in enumerate(few_shot_task.choices):
                option = chr(ord("A") + i)
                prompt += f"({option}) {choice}" + "\n"

                if choice == few_shot_task.answer:
                    multichoice_answer = f"{option})"

            assert (
                multichoice_answer is not None
            ), "Answer must appear in list of choices"
            assert multichoice_answer in prompt, "Answer must appear in prompt"

            prompt += "Answer: (" + multichoice_answer
            prompt += task_seperator

    problem = binary_classification_task_to_problem(
        task=task.classification_task,
        example_to_problem_fn=example_to_problem_fn,
        example_seperator=example_seperator,
    )

    prompt += problem.prompt + problem.expected_completion
    prompt += example_seperator
    prompt += question + "\n"
    prompt += "Choices:" + "\n"

    multichoice_answer = None

    for i, choice in enumerate(task.choices):
        option = chr(ord("A") + i)
        prompt += f"({option}) {choice}" + "\n"

        if choice == task.answer:
            multichoice_answer = f"{option}"

    assert multichoice_answer is not None, "Answer must appear in list of choices"

    prompt += "Answer: ("

    return Problem(
        prompt=prompt,
        expected_completion=multichoice_answer,
    )


# %%


def get_french_words(fname: str = "datasets/french-words.txt") -> Tuple[str, ...]:
    """Returns a list of French words."""
    assert os.path.isfile(fname), "French words dataset not found."
    return tuple(open(fname, "r").read().splitlines())


# %%


def get_news_headlines_dataset(return_features=False):
    """Returns a list of news headlines (and optionally features to use for crisp classification)."""
    assert os.path.isfile("datasets/news.jsonl"), (
        "News headlines dataset not found. Please download dataset at"
        " `https://www.kaggle.com/datasets/rmisra/news-category-dataset?resource=download`"
        " and place it in `datasets/news.jsonl`."
    )

    jsonls = open("datasets/news.jsonl", "r").read().splitlines()
    jsonls = [json.loads(s) for s in jsonls]
    headlines = [jl["headline"] for jl in jsonls if len(jl["headline"]) > 0]
    headlines = sorted(random.sample(headlines, k=100_000))
    headline_words = set(w for h in headlines for w in h.split())
    words = [
        w for w in open("datasets/words.txt").read().splitlines() if w in headline_words
    ]

    ftoi = {}

    for w in words:
        feature = w

        for h in headlines:
            if f" {w.lower()} " not in h.lower():
                continue  # negative example

            ftoi[feature] = ftoi.get(feature, 0) + 1

    features = list(
        map(
            lambda kv: kv[0],
            sorted(
                filter(lambda kv: kv[1] >= 10 and len(kv[0]) >= 3, ftoi.items()),
                key=lambda kv: kv[1],
            ),
        )
    )  # features are at least 3 chars long and appear in at least 10 headlines

    if return_features:
        return headlines, features

    return headlines


# %%


@functools.lru_cache(maxsize=1)
def get_contrived_features_dataset(
    max_features: Optional[int] = None,
    shuffled: bool = False,
    seed: Optional[int] = None,
) -> Tuple[str]:
    if seed is not None:
        random.seed(seed)

    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokens = list(tokenizer.get_vocab().keys())

    root = utils.TrieNode()

    for w in tokens:
        root.insert(w)

    leaves = utils.find_leaves(root)
    dictionary = set(open("datasets/words.txt", "r").read().splitlines())
    words = [w for w in dictionary if w in leaves]
    n_features = len(words)

    if max_features:
        n_features = min(max_features, len(words))

    assert n_features <= len(
        words
    ), f"Cannot have more features than words in dictionary."

    words = random.sample(words, k=n_features)

    if not shuffled:
        words = sorted(words)

    return tuple(words)


# %%


def get_common_words_features_dataset(
    max_features: Optional[int] = None,
    shuffled: bool = False,
    seed: Optional[int] = None,
    all_lowercase: bool = True,
    min_length: int = 0,
    filter_out_subsets: bool = True,
) -> Tuple[str, ...]:
    assert min_length >= 0, "`min_length` must be non-negative."

    if seed is not None:
        random.seed(seed)

    with open("datasets/1k-most-common-words.txt", "r") as f:
        words = f.read().splitlines()

    words = [w for w in words if w.isalpha()]
    words = [w for w in words if len(w) >= min_length]

    if filter_out_subsets:
        words = tuple(utils.filter_subsets(words))

    words = sorted(words)

    if max_features:
        words = words[:max_features]

    if all_lowercase:
        words = [w.lower() for w in words]

    if shuffled:
        words = random.sample(words, k=len(words))

    words = tuple(words)

    if max_features:
        assert len(words) == max_features, "Must have correct number of features."

    if shuffled:
        assert words != sorted(words), "Must be shuffled."

    assert all(len(w) >= 1 for w in words), "Words must be at least 1 character long."
    assert all(w.isalpha() for w in words), "Words must be alphabetical."

    return words


######################
# HIGHER-ORDER RULES #
######################


class ConjunctionRule(Rule):
    def __init__(self, explanation: str, rules: Tuple[Rule]) -> None:
        super().__init__(
            explanation=explanation,
            generation_behaviour=BooleanOperationGenerationBehaviour(rules, all),
            discrimination_behaviour=BinaryOperatorDiscriminationBehaviour(rules, all),
        )


class DisjunctionRule(Rule):
    def __init__(self, explanation: str, rules: Tuple[Rule]) -> None:
        super().__init__(
            explanation=explanation,
            generation_behaviour=BooleanOperationGenerationBehaviour(rules, any),
            discrimination_behaviour=BinaryOperatorDiscriminationBehaviour(rules, any),
        )


class XorRule(Rule):
    def __init__(self, explanation: str, rules: Tuple[Rule]) -> None:
        xor = utils.xor
        super().__init__(
            explanation=explanation,
            generation_behaviour=BooleanOperationGenerationBehaviour(rules, xor),
            discrimination_behaviour=BinaryOperatorDiscriminationBehaviour(rules, xor),
        )


class NegationRule(Rule):
    def __init__(self, explanation: str, rule: Rule) -> None:
        super().__init__(
            explanation=explanation,
            generation_behaviour=NegationRuleGenerationBehaviour(rule),
            discrimination_behaviour=NegationRuleDiscriminationBehaviour(rule),
        )


##############
# GENERATORS #
##############


class NegationRuleGenerationBehaviour(BinaryClassificationExampleGenerationBehaviour):
    def __init__(self, rule: Rule) -> None:
        assert isinstance(rule, Rule), "Rule must be of type `Rule`"
        self.rule = rule

    def generate_passing_example(self) -> BinaryClassificationExample:
        example = self.rule.generate_failing_example()

        negated_example = BinaryClassificationExample(
            value=example.value,
            label=not example.label,
        )

        return negated_example

    def generate_failing_example(self) -> BinaryClassificationExample:
        example = self.rule.generate_passing_example()

        negated_example = BinaryClassificationExample(
            value=example.value,
            label=not example.label,
        )

        return negated_example


class BooleanOperationGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    """Attempts to generate examples that satisfy an operator."""

    def __init__(
        self,
        rules: Tuple[Rule],
        operator: Callable[[bool], bool],
        max_attempts: int = 3_628_800,  # 10!
    ):
        assert (
            len(rules) >= 1
        ), "Must have at least one rule to generate conjunction rule"
        assert all(
            isinstance(r, Rule) for r in rules
        ), "All rules must be of type `Rule`"
        assert max_attempts >= 1, "Must have at least one attempt to generate example"

        self.rules = rules
        self.operator = operator
        self.max_attempts = max_attempts

    def generate_passing_example(self) -> BinaryClassificationExample:
        sampled = [r.generate_passing_example() for r in self.rules]

        assert 1 <= len(sampled) == len(self.rules), "Must have one example per rule"
        assert all(
            isinstance(s, BinaryClassificationExample) for s in sampled
        ), "All rules must generate BinaryClassificationExamples"
        assert all(s.label for s in sampled), "All rules must generate passing examples"
        assert all(
            isinstance(s.value, str) for s in sampled
        ), "All rules must generate examples with string values"
        assert all(
            len(s.value) > 0 for s in sampled
        ), "All rules must generate examples with non-empty string values"

        words = " ".join(s.value for s in sampled).split(" ")
        words = random.sample(words, k=len(words))

        n_words_per_example = len(sampled[0].value.split(" "))

        assert len(words) == n_words_per_example * len(
            sampled
        ), "Expected to have `n_words_per_example` words per example"

        for i, candidate_words in enumerate(
            itertools.permutations(words, r=n_words_per_example)
        ):
            candidate_example = BinaryClassificationExample(
                value=" ".join(candidate_words),
                label=True,
            )

            if self.operator([r(candidate_example) for r in self.rules]):
                assert isinstance(
                    candidate_example, BinaryClassificationExample
                ), "Must generate BinaryClassificationExample"
                assert candidate_example.label, "Must generate passing example"
                assert isinstance(
                    candidate_example.value, str
                ), "Must generate example with string value"
                assert (
                    len(candidate_example.value) > 0
                ), "Must generate example with non-empty string value"
                assert (
                    len(candidate_example.value.split(" ")) == n_words_per_example
                ), "Must have `n_words_per_example` words per example"

                return candidate_example

            if i + 1 >= self.max_attempts:
                break

        raise RuntimeError(
            f"Failed to generate passing example after {self.max_attempts:,} attempts"
        )

    def generate_failing_example(self) -> BinaryClassificationExample:
        sampled = [r.generate_passing_example() for r in self.rules]  # for variety
        # sampled += [r.generate_failing_example() for r in self.rules]

        assert (
            1 <= len(self.rules) <= len(sampled)
        ), "Must have at least one example per rule"
        assert all(
            isinstance(s, BinaryClassificationExample) for s in sampled
        ), "All rules must generate BinaryClassificationExamples"
        assert all(
            isinstance(s.value, str) for s in sampled
        ), "All rules must generate examples with string values"
        assert all(
            len(s.value) > 0 for s in sampled
        ), "All rules must generate examples with non-empty string values"

        words = " ".join(s.value for s in sampled).split(" ")
        words = random.sample(words, k=len(words))

        n_words_per_example = len(sampled[0].value.split(" "))

        assert len(words) == n_words_per_example * len(
            sampled
        ), "Expected to have `n_words_per_example` words per example"

        for i, candidate_words in enumerate(
            itertools.permutations(words, r=n_words_per_example)
        ):
            candidate_example = BinaryClassificationExample(
                value=" ".join(candidate_words),
                label=False,
            )

            if not self.operator([r(candidate_example) for r in self.rules]):
                assert isinstance(
                    candidate_example, BinaryClassificationExample
                ), "Must generate BinaryClassificationExample"
                assert not candidate_example.label, "Must generate failing example"
                assert isinstance(
                    candidate_example.value, str
                ), "Must generate example with string value"
                assert (
                    len(candidate_example.value) > 0
                ), "Must generate example with non-empty string value"
                assert (
                    len(candidate_example.value.split(" ")) == n_words_per_example
                ), "Must have `n_words_per_example` words per example"

                return candidate_example

            if i + 1 >= self.max_attempts:
                break

        raise RuntimeError(
            f"Failed to generate failing example after {self.max_attempts:,} attempts"
        )


# %%


class ContainsFeatureGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        feature: str,
        words: Sequence[str],
        n_words_per_example: int,
        allowed_feature_idxs: Optional[Sequence[int]] = None,
    ):
        if allowed_feature_idxs is None:
            allowed_feature_idxs = tuple(range(n_words_per_example))

        assert n_words_per_example >= 1, "Must have at least one word per example"
        assert feature in words, f"Feature `{feature}` must be in list of valid `words`"
        assert len(words) >= n_words_per_example + 1, (
            "Must have at least `n_words_per_example + 1` words to generate failing"
            " example"
        )
        assert (
            len(allowed_feature_idxs) > 0
        ), "Must have at least one allowed feature index"
        assert all(
            0 <= i < n_words_per_example for i in allowed_feature_idxs
        ), "All allowed feature indices must be in range"

        self.feature = feature
        self.words = words
        self.n_words_per_example = n_words_per_example
        self.allowed_feature_idxs = allowed_feature_idxs

    def generate_passing_example(self):
        other_words = iter(
            utils.sample(
                self.words, k=self.n_words_per_example - 1, exclude={self.feature}
            )
        )
        feature_idx = random.choice(self.allowed_feature_idxs)
        sampled = [
            next(other_words) if i != feature_idx else self.feature
            for i in range(self.n_words_per_example)
        ]

        assert sampled[feature_idx] == self.feature, (
            f"Feature `{self.feature}` must appear in all allowed feature indices"
            f" `{feature_idx}`"
        )
        assert all(
            sampled[i] != self.feature
            for i in range(self.n_words_per_example)
            if i != feature_idx
        ), (
            f"Feature `{self.feature}` can only appear in allowed feature indices"
            f" `{feature_idx}`"
        )
        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(value=" ".join(sampled), label=True)

    def generate_failing_example(self):
        sampled = utils.sample(
            self.words, k=self.n_words_per_example, exclude={self.feature}
        )

        forbidden_feature_idxs = tuple(
            i
            for i in range(self.n_words_per_example)
            if i not in self.allowed_feature_idxs
        )

        if random.random() < 0.5 and len(forbidden_feature_idxs) >= 1:
            feature_idx = random.choice(forbidden_feature_idxs)
            sampled = (
                sampled[:feature_idx] + (self.feature,) + sampled[feature_idx + 1 :]
            )

        assert all(
            [sampled[i] != self.feature for i in self.allowed_feature_idxs]
        ), f"Feature `{self.feature}` must not appear in valid place in failing example"
        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(value=" ".join(sampled), label=False)


# %%


class DoesNotContainFeatureGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        feature: str,
        words: Sequence[str],
        n_words_per_example: int,
    ):
        assert n_words_per_example >= 1, "Must have at least one word per example"
        assert feature in words, f"Feature `{feature}` must be in list of valid `words`"
        assert len(words) >= n_words_per_example + 1, (
            "Must have at least `n_words_per_example + 1` words to generate failing"
            " example"
        )

        self.features = words
        self.anti_feature = feature
        self.n_words_per_example = n_words_per_example

    def generate_passing_example(self):
        sampled = utils.sample(
            self.features,
            k=self.n_words_per_example,
            exclude={self.anti_feature},
        )

        assert all(
            w in self.features for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"
        assert (
            self.anti_feature not in sampled
        ), "Expected feature to not appear in passing example."

        return BinaryClassificationExample(value=" ".join(sampled), label=True)

    def generate_failing_example(self):
        sampled = utils.sample(
            self.features,
            k=self.n_words_per_example,
            include={self.anti_feature},
        )

        assert all(
            w in self.features for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"
        assert (
            self.anti_feature in sampled
        ), "Expected feature to not appear in passing example."

        return BinaryClassificationExample(value=" ".join(sampled), label=False)


# %%


class SampledWordsContainAllFeaturesGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        features: Sequence[str],
        words: Sequence[str],
        n_words_per_example: int,
    ):
        assert n_words_per_example >= 1, "Must have at least one word per example"
        assert len(features) >= 1, "Must have at least one feature"
        assert (
            len(features) <= n_words_per_example
        ), "Cannot have more features than words"
        assert len(words) >= n_words_per_example + len(features), (
            "Must have at least `n_words_per_example + n_features` words to generate"
            " failing example"
        )

        self.features = set(features)
        self.words = words
        self.n_words_per_example = n_words_per_example

    def generate_passing_example(self):
        sampled = utils.sample(
            self.words,
            k=self.n_words_per_example,
            include=set(
                self.features,
            ),
        )

        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"
        assert all(
            f in sampled for f in self.features
        ), f"All features must appear in passing example"

        return BinaryClassificationExample(value=" ".join(sampled), label=True)

    def generate_failing_example(self):
        sampled = utils.sample(
            self.words,
            k=self.n_words_per_example,
        )

        if all(f in sampled for f in self.features):
            random_f = random.choice(tuple(self.features))
            count_f = sampled.count(random_f)
            sampled = tuple(w for w in sampled if w != random_f)
            sampled += utils.sample(self.words, k=count_f, exclude=self.features)

        assert not all(
            f in sampled for f in self.features
        ), f"All features must not appear in failing example"
        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(value=" ".join(sampled), label=False)


# %%


class SampledWordsContainAnySpaceSeparatedFeaturesGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        features: Sequence[str],
        words: Sequence[str],
        n_words_per_example: int,
    ):
        assert n_words_per_example >= 1, "Must have at least one word per example"
        assert len(features) >= 1, "Must have at least one feature"
        assert len(words) >= n_words_per_example + 1, (
            "Must have at least `n_words_per_example + 1` words to generate failing"
            " example"
        )
        assert all(f in words for f in features), "All features must be in `words`"

        self.features = features
        self.words = words
        self.n_words_per_example = n_words_per_example

    def generate_passing_example(self):
        feature = random.choice(tuple(self.features))

        sampled = utils.sample(
            self.words,
            k=self.n_words_per_example,
            include={feature},
        )

        assert any(
            f in sampled for f in self.features
        ), f"At least one feature must appear in passing example"
        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(value=" ".join(sampled), label=True)

    def generate_failing_example(self):
        sampled = utils.sample(
            self.words,
            k=self.n_words_per_example,
            exclude=self.features,
        )

        assert all(
            f not in sampled for f in self.features
        ), f"No features cam appear in failing example"
        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(value=" ".join(sampled), label=False)


# %%


class SampledWordsAppearCertainNumberOfTimesGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        feature: str,
        words: Sequence[str],
        n_words_per_example: int,
        min_feature_count: int = 1,
        max_feature_count: int = 1,
    ):
        assert 1 <= min_feature_count <= n_words_per_example, (
            "Feature must appear at least once but no more than the number of words per"
            " example"
        )
        assert min_feature_count <= max_feature_count <= n_words_per_example, (
            "Feature can only appear between `min_feature_count` and"
            " `n_words_per_example` times in each example"
        )

        self.feature = feature
        self.words = words
        self.n_words_per_example = n_words_per_example
        self.min_feature_count = min_feature_count
        self.max_feature_count = max_feature_count

    def generate_passing_example(self):
        k = random.randint(self.min_feature_count, self.max_feature_count)
        other_words = utils.sample(
            self.words, k=self.n_words_per_example - k, exclude={self.feature}
        )
        sampled = random.sample(
            (self.feature,) * k + other_words, k=self.n_words_per_example
        )

        assert self.min_feature_count <= k <= self.max_feature_count, (
            f"Feature `{self.feature}` must appear between {self.min_feature_count} and"
            f" {self.max_feature_count} times in each example"
        )
        assert sampled.count(self.feature) == k, (
            f"Expected `{self.feature}` to appear exactly {k} times in example but got"
            f" {sampled}"
        )

        sampled = " ".join(sampled)

        return BinaryClassificationExample(value=sampled, label=True)

    def generate_failing_example(self):
        k = random.choice(
            [
                i
                for i in range(self.n_words_per_example)
                if i < self.min_feature_count or i > self.max_feature_count
            ]
        )
        other_words = utils.sample(
            self.words, k=self.n_words_per_example - k, exclude={self.feature}
        )
        sampled = random.sample(
            (self.feature,) * k + other_words, k=self.n_words_per_example
        )

        assert k < self.min_feature_count or k > self.max_feature_count, (
            f"Feature `{self.feature}` can't appear between"
            f" {self.min_feature_count} and {self.max_feature_count} times in a failing"
            " example"
        )
        assert sampled.count(self.feature) == k, (
            f"Expected `{self.feature}` to appear exactly {k} times in example but got"
            f" {sampled}"
        )

        sampled = " ".join(sampled)

        return BinaryClassificationExample(value=sampled, label=False)


# %%


class ContainsCertainNumberOfWordsInExampleGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        words: Sequence[str],
        min_n_words_per_example: int,
        max_n_words_per_example: int,
        n_words_in_passing_example: int,
    ):
        assert min_n_words_per_example >= 1, "Must have at least one word per example"
        assert (
            min_n_words_per_example < max_n_words_per_example
        ), "`min_n_words_per_example` must be less than `max_n_words_per_example`"
        assert (
            min_n_words_per_example
            <= n_words_in_passing_example
            <= max_n_words_per_example
        ), (
            "Number of words in passing example must be between"
            " `min_n_words_per_example` and `max_n_words_per_example`"
        )
        assert len(words) >= max_n_words_per_example, (
            "Must have at least `max_n_words_per_example` words in `words` to sample"
            " without replacement"
        )

        self.words = words
        self.min_n_words_per_example = min_n_words_per_example
        self.max_n_words_per_example = max_n_words_per_example
        self.n_words_in_passing_example = n_words_in_passing_example

    def generate_passing_example(self):
        sampled = random.sample(self.words, k=self.n_words_in_passing_example)

        assert len(sampled) == self.n_words_in_passing_example, (
            f"Must have exactly {self.n_words_in_passing_example} words in passing"
            " example"
        )

        sampled = " ".join(sampled)
        return BinaryClassificationExample(value=sampled, label=True)

    def generate_failing_example(self):
        k = random.choice(
            [
                i
                for i in range(
                    self.min_n_words_per_example, self.max_n_words_per_example + 1
                )
                if i != self.n_words_in_passing_example
            ]
        )
        sampled = random.sample(self.words, k=k)

        assert (
            self.min_n_words_per_example <= len(sampled) <= self.max_n_words_per_example
        ), (
            f"Must have between {self.min_n_words_per_example} and"
            f" {self.max_n_words_per_example} words per example"
        )
        assert (
            len(sampled) != self.n_words_in_passing_example
        ), f"Can't have {self.n_words_in_passing_example} words in failing example"

        sampled = " ".join(sampled)
        return BinaryClassificationExample(value=sampled, label=False)


# %%


class SampledWordsAreCertainCaseGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        words: Sequence[str],
        n_words_per_example: int,
        upper_case_is_passing: bool = True,
    ):
        assert n_words_per_example >= 1, "Must have at least one word per example"
        assert (
            len(words) >= n_words_per_example
        ), "Must have at least as many words as words per example"

        self.words = words
        self.n_words_per_example = n_words_per_example
        self.upper_case_is_passing = upper_case_is_passing

    def generate_passing_example(self):
        sampled = " ".join(random.sample(self.words, k=self.n_words_per_example))
        sampled = sampled.upper() if self.upper_case_is_passing else sampled.lower()

        assert len(sampled) > 0, "Must have at least one character per example"

        if self.upper_case_is_passing:
            assert (
                sampled.isupper()
            ), f"Expected passing example `{sampled}` to be upper case"

        if not self.upper_case_is_passing:
            assert (
                sampled.islower()
            ), f"Expected passing example `{sampled}` to be lower case"

        return BinaryClassificationExample(value=sampled, label=True)

    def generate_failing_example(self):
        sampled = " ".join(random.sample(self.words, k=self.n_words_per_example))
        sampled = sampled.lower() if self.upper_case_is_passing else sampled.upper()

        assert len(sampled) > 0, "Must have at least one character per example"

        if self.upper_case_is_passing:
            assert (
                sampled.islower()
            ), f"Expected failing example `{sampled}` to be upper case"

        if not self.upper_case_is_passing:
            assert (
                sampled.isupper()
            ), f"Expected failing example `{sampled}` to be upper case"

        return BinaryClassificationExample(value=sampled, label=False)


# %%


class SampledWordsAreInFormOfAQuestionGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        words: Sequence[str],
        n_words_per_example: int,
        question_is_passing: bool = True,
    ):
        assert n_words_per_example >= 1, "Must have at least one word per example"
        assert (
            len(words) >= n_words_per_example
        ), "Must have at least as many words as words per example"

        self.words = words
        self.n_words_per_example = n_words_per_example
        self.question_is_passing = question_is_passing
        self.question_is_failing = (
            not question_is_passing
        )  # makes code below more readable

    def generate_passing_example(self):
        sampled = random.sample(self.words, k=self.n_words_per_example)

        assert all(
            w in self.words for w in sampled
        ), f"Expected all words `{sampled}` to be in list of valid words"

        sampled = " ".join(sampled)

        if self.question_is_passing:
            sampled += "?"

        if self.question_is_passing:
            assert sampled.endswith(
                "?"
            ), f"Expected passing example `{sampled}` to be a question"

        if self.question_is_failing:
            assert not sampled.endswith(
                "?"
            ), f"Expected passing example `{sampled}` to not be a question"

        return BinaryClassificationExample(value=sampled, label=True)

    def generate_failing_example(self):
        sampled = random.sample(self.words, k=self.n_words_per_example)

        assert all(
            w in self.words for w in sampled
        ), f"Expected all words `{sampled}` to be in list of valid words"

        sampled = " ".join(sampled)

        if self.question_is_failing:
            sampled += "?"

        if self.question_is_passing:
            assert not sampled.endswith(
                "?"
            ), f"Expected failing example `{sampled}` to not be a question"

        if self.question_is_failing:
            assert sampled.endswith(
                "?"
            ), f"Expected failing example `{sampled}` to be a question"

        return BinaryClassificationExample(value=sampled, label=False)


# %%


class SampledWordsAreSortedGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        words: Sequence[str],
        n_words_per_example: int,
        is_descending: bool = False,
    ):
        assert (
            n_words_per_example >= 2
        ), "Must have at least one word per example so sorting behaviour can be seen"
        assert len(set(words)) >= 2, (
            "Must have at least two unique words to sort so sorting behaviour can be"
            " seen"
        )
        assert (
            len(words) >= n_words_per_example
        ), "Must have at least as many words as words per example"

        self.words = words
        self.n_words_per_example = n_words_per_example
        self.is_descending = is_descending

    def generate_passing_example(self):
        sampled = sorted(
            random.sample(self.words, k=self.n_words_per_example),
            reverse=self.is_descending,
        )

        assert (
            len(sampled) == self.n_words_per_example
        ), "Sample must contain `n_words_per_example` words"
        assert (
            len(set(sampled)) >= 2
        ), "Sample must contain at least two unique words to observe sorting behaviour"
        assert sampled == sorted(
            sampled, reverse=self.is_descending
        ), "Sample must be sorted"
        assert all(
            w in self.words for w in sampled
        ), "Sample must only contain words from `words`"

        return BinaryClassificationExample(
            value=" ".join(sampled),
            label=True,
        )

    def generate_failing_example(self):
        sampled = random.sample(self.words, k=self.n_words_per_example)

        while True:
            sampled = random.sample(self.words, k=self.n_words_per_example)

            if len(set(sampled)) == 1:
                continue  # draw another sample since sorting behaviour can't be observed with single word

            if sampled == sorted(sampled, reverse=self.is_descending):
                sampled = sampled[::-1]

            break

        assert (
            len(sampled) == self.n_words_per_example
        ), "Sample must contain `n_words_per_example` words"
        assert (
            len(set(sampled)) >= 2
        ), "Sample must contain at least two unique words to observe sorting behaviour"
        assert sampled != sorted(
            sampled, reverse=self.is_descending
        ), "Sample cannot be sorted in correct order"
        assert all(
            w in self.words for w in sampled
        ), "Sample must only contain words from `words`"

        return BinaryClassificationExample(
            value=" ".join(sampled),
            label=False,
        )


# %%


# Ordered means that only the words in the sequence matter.
# E.g, A, B could be C, A, B, E, D but sorted demands A, B, C, D, E.
class WordsAreOrderedGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        features: Sequence[str],
        words: Sequence[str],
        n_words_per_example: int,
        is_descending: bool = False,
    ):
        assert (
            len(features) >= 2
        ), "Must have at least two features (else is always True)"
        assert (
            len(features) <= n_words_per_example
        ), "Must have at least as many words as features"
        assert (
            n_words_per_example >= 2
        ), "Must have at least one word per example so sorting behaviour can be seen"
        assert len(set(words)) >= 2, (
            "Must have at least two unique words to sort so sorting behaviour can be"
            " seen"
        )
        assert (
            len(words) >= n_words_per_example
        ), "Must have at least as many words as words per example"

        self.words = words
        self.features = features
        self.n_words_per_example = n_words_per_example
        self.is_descending = is_descending

    def generate_passing_example(self):
        sampled = utils.sample(
            self.words, k=self.n_words_per_example, exclude=set(self.features)
        )

        sorted_features = sorted(self.features, reverse=self.is_descending)

        feature_idxs = sorted(
            random.sample(range(self.n_words_per_example), k=len(self.features))
        )

        for i, feature in zip(feature_idxs, sorted_features):
            sampled = sampled[:i] + (feature,) + sampled[i + 1 :]

        assert (
            len(sampled) == self.n_words_per_example
        ), "Sample must contain `n_words_per_example` words"
        assert (
            len(set(sampled)) >= 2
        ), "Sample must contain at least two unique words to observe sorting behaviour"
        assert [
            sampled.index(f) for f in sorted(self.features, reverse=self.is_descending)
        ] == sorted(
            [
                sampled.index(f)
                for f in sorted(self.features, reverse=self.is_descending)
            ]
        ), "Sample must be ordered"
        assert all(
            w in self.words for w in sampled
        ), "Sample must only contain words from `words`"

        return BinaryClassificationExample(
            value=" ".join(sampled),
            label=True,
        )

    def generate_failing_example(self):
        sampled = utils.sample(
            self.words, k=self.n_words_per_example, exclude=set(self.features)
        )

        jumlbed_features = random.sample(self.features, k=len(self.features))

        if jumlbed_features == sorted(self.features, reverse=self.is_descending):
            jumlbed_features = jumlbed_features[::-1]

        feature_idxs = sorted(
            random.sample(range(self.n_words_per_example), k=len(self.features))
        )

        for i, feature in zip(feature_idxs, jumlbed_features):
            sampled = sampled[:i] + (feature,) + sampled[i + 1 :]

        assert (
            len(sampled) == self.n_words_per_example
        ), "Sample must contain `n_words_per_example` words"
        assert (
            len(set(sampled)) >= 2
        ), "Sample must contain at least two unique words to observe sorting behaviour"
        assert [
            sampled.index(f) for f in sorted(self.features, reverse=self.is_descending)
        ] != sorted(
            [
                sampled.index(f)
                for f in sorted(self.features, reverse=self.is_descending)
            ]
        ), "Sample must not be ordered"
        assert all(
            w in self.words for w in sampled
        ), "Sample must only contain words from `words`"

        return BinaryClassificationExample(
            value=" ".join(sampled),
            label=False,
        )


# %%


class SampledWordsAreWithinXPositionsOfEachOtherGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        w: str,
        w_prime: str,
        words: Sequence[str],
        x: int,
        n_words_per_example: int,
    ):
        assert (
            n_words_per_example >= 2
        ), "Must have at least one word per example so sorting behaviour can be seen"
        assert (
            len(words) >= n_words_per_example
        ), "Must have at least as many words as words per example"
        assert x >= 1, "Must have at least one position difference between words"
        assert x < (n_words_per_example // 2), (
            f"{x=} must be less than half the number of words per example"
            f" {n_words_per_example=} to gaurantee failure generation"
        )

        self.words = words
        self.w = w
        self.w_prime = w_prime
        self.x = x
        self.n_words_per_example = n_words_per_example

    def generate_passing_example(self):
        sampled = utils.sample(
            self.words,
            k=self.n_words_per_example,
            exclude={self.w, self.w_prime},
        )

        w_idx = random.randrange(0, len(sampled))
        low = max(0, w_idx - self.x)
        high = min(len(sampled), w_idx + self.x + 1)
        (w_prime_idx,) = utils.sample(range(low, high), exclude={w_idx})

        assert 0 <= w_idx < len(sampled), "w_idx must be within sampled"
        assert 0 <= w_prime_idx < len(sampled), "w_prime_idx must be within sampled"
        assert (
            abs(w_idx - w_prime_idx) <= self.x
        ), f"w and w_prime must be within {self.x} positions of each other"

        fst_idx = min(w_idx, w_prime_idx)
        snd_idx = max(w_idx, w_prime_idx)

        sampled = (
            sampled[:fst_idx]
            + (self.w,)
            + sampled[fst_idx + 1 : snd_idx]
            + (self.w_prime,)
            + sampled[snd_idx + 1 :]
        )

        assert (
            len(sampled) == self.n_words_per_example
        ), f"Sample must contain `n_words_per_example` words but got {len(sampled)}"
        assert (
            abs(sampled.index(self.w) - sampled.index(self.w_prime)) <= self.x
        ), "Features must be within X positions of each other"

        return BinaryClassificationExample(
            value=" ".join(sampled),
            label=True,
        )

    def generate_failing_example(self):
        sampled = utils.sample(
            self.words,
            k=self.n_words_per_example,
            exclude={self.w, self.w_prime},
        )

        w_idx = random.randrange(0, len(sampled))
        low = max(0, w_idx - self.x)
        high = min(len(sampled), w_idx + self.x + 1)
        (w_prime_idx,) = utils.sample(
            range(len(sampled)), exclude=set(range(low, high))
        )

        assert 0 <= w_idx < len(sampled), "w_idx must be within sampled"
        assert 0 <= w_prime_idx < len(sampled), "w_prime_idx must be within sampled"
        assert (
            abs(w_idx - w_prime_idx) > self.x
        ), f"w and w_prime must be at least {self.x} positions apart"

        fst_idx = min(w_idx, w_prime_idx)
        snd_idx = max(w_idx, w_prime_idx)

        sampled = (
            sampled[:fst_idx]
            + (self.w,)
            + sampled[fst_idx + 1 : snd_idx]
            + (self.w_prime,)
            + sampled[snd_idx + 1 :]
        )

        assert (
            len(sampled) == self.n_words_per_example
        ), f"Sample must contain `n_words_per_example` words but got {len(sampled)}"
        assert (
            abs(sampled.index(self.w) - sampled.index(self.w_prime)) > self.x
        ), "Features must not be within X positions of each other"

        return BinaryClassificationExample(
            value=" ".join(sampled),
            label=False,
        )


# %%


class SampledSentencesContainFeatureGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self, feature: str, sentences: Sequence[str], ignore_case: bool = False
    ):
        assert len(feature) > 0, "Feature can't be the empty string"
        assert len(sentences) > 0, "Must have at least one sentence"

        sentences_with_feature = [
            sentence for sentence in sentences if feature in sentence
        ]
        sentences_without_feature = [
            sentence for sentence in sentences if feature not in sentence
        ]

        if ignore_case:
            sentences_with_feature = [
                sentence
                for sentence in sentences
                if feature.lower() in sentence.lower()
            ]
            sentences_without_feature = [
                sentence
                for sentence in sentences
                if feature.lower() not in sentence.lower()
            ]

        assert (
            len(sentences_with_feature) > 0
        ), f"Feature `{feature}` must be appear in at least one sentence"
        assert (
            len(sentences_without_feature) > 0
        ), f"Feature `{feature}` must not appear in all sentences"

        self.word = feature
        self.words = sentences
        self.sentences_with_feature = sentences_with_feature
        self.sentences_without_feature = sentences_without_feature

    def generate_passing_example(self):
        [passing_sentence] = random.sample(self.sentences_with_feature, k=1)
        return BinaryClassificationExample(
            value=passing_sentence,
            label=True,
        )

    def generate_failing_example(self):
        [failing_sentence] = random.sample(self.sentences_without_feature, k=1)
        return BinaryClassificationExample(
            value=failing_sentence,
            label=False,
        )


# %%


class SampleFromConjunctionOfDisjunctionsGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        conjunction_of_disjunction_of_features: Sequence[Sequence[str]],
        words: Sequence[str],
        n_words_per_example: int,
    ):
        assert (
            len(conjunction_of_disjunction_of_features) >= 1
        ), "Must have at least one disjunction of features"
        assert all(
            len(fs) >= 1 for fs in conjunction_of_disjunction_of_features
        ), "Must have at least one feature in each disjunction"
        assert all(
            all(map(lambda f: len(f) > 0, fs))
            for fs in conjunction_of_disjunction_of_features
        ), "Feature can't be the empty string"
        assert (
            len(conjunction_of_disjunction_of_features) <= n_words_per_example
        ), "Can't have more features than words per example"
        assert len(words) >= n_words_per_example + len(
            conjunction_of_disjunction_of_features
        ), "Must have enough words to create failing example"

        self.cd_features = conjunction_of_disjunction_of_features
        self.flattened_features = set(
            itertools.chain.from_iterable(conjunction_of_disjunction_of_features)
        )
        self.words = words
        self.n_words_per_example = n_words_per_example

    def generate_passing_example(self):
        features = tuple(random.choice(fs) for fs in self.cd_features)
        other_words = utils.sample(
            self.words,
            k=self.n_words_per_example - len(features),
            exclude=self.flattened_features,
        )
        sampled = random.sample(features + other_words, k=self.n_words_per_example)

        assert all(
            any(set(sampled).intersection(fs)) for fs in self.cd_features
        ), "Sample must contain at least one feature from each disjunction"
        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(
            value=" ".join(sampled),
            label=True,
        )

    def generate_failing_example(self):
        def _sample():
            return random.sample(self.words, k=self.n_words_per_example)

        while all(
            any(set(sampled := _sample()).intersection(fs)) for fs in self.cd_features
        ):
            sampled = _sample()

        assert not all(
            any(set(sampled).intersection(fs)) for fs in self.cd_features
        ), "Sample must not contain any feature from each disjunction"
        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(
            value=" ".join(sampled),
            label=False,
        )


# %%


class ContainsWordFollowedByAnotherWordGenerationBehaviour(
    BinaryClassificationExampleGenerationBehaviour
):
    def __init__(
        self,
        first_feature: str,
        second_feature: str,
        words: Sequence[str],
        n_words_per_example: int,
    ):
        assert n_words_per_example >= 1, "Must have at least one word per example"
        assert (
            first_feature in words
        ), f"Feature `{first_feature}` must be in list of valid `words`"
        assert (
            second_feature in words
        ), f"Feature `{second_feature}` must be in list of valid `words`"
        assert len(words) >= n_words_per_example + 2, (
            "Must have at least `n_words_per_example + 2` words to generate failing"
            " example"
        )

        self.first_feature = first_feature
        self.second_feature = second_feature
        self.words = words
        self.n_words_per_example = n_words_per_example

    def generate_passing_example(self):
        other_words = utils.sample(
            self.words,
            k=self.n_words_per_example,
            exclude={self.first_feature, self.second_feature},
        )

        idxs = random.sample(range(self.n_words_per_example), k=2)
        fst_idx, snd_idx = min(idxs), max(idxs)

        sampled = (
            other_words[:fst_idx]
            + (self.first_feature,)
            + other_words[fst_idx + 1 : snd_idx]
            + (self.second_feature,)
            + other_words[snd_idx + 1 :]
        )

        assert (
            self.first_feature in sampled
        ), f"First feature must appear in passing example"
        assert (
            self.second_feature in sampled
        ), f"Second feature must appear in passing example"
        assert sampled.index(self.first_feature) < sampled.index(
            self.second_feature
        ), f"First feature must appear before second feature"
        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(value=" ".join(sampled), label=True)

    def generate_failing_example(self):
        sampled = utils.sample(
            self.words,
            k=self.n_words_per_example,
            exclude={self.first_feature, self.second_feature},
        )

        if random.random() < 0.5:
            idxs = random.sample(range(self.n_words_per_example), k=2)
            fst_idx, snd_idx = min(idxs), max(idxs)

            sampled = (
                sampled[:fst_idx]
                + (self.second_feature,)
                + sampled[fst_idx + 1 : snd_idx]
                + (self.first_feature,)
                + sampled[snd_idx + 1 :]
            )

        if self.first_feature in sampled and self.second_feature in sampled:
            assert sampled.index(self.first_feature) > sampled.index(
                self.second_feature
            ), f"First feature must appear after second feature"

        assert all(
            w in self.words for w in sampled
        ), f"All sampled words must be in list of valid `words`"
        assert (
            len(sampled) == self.n_words_per_example
        ), f"Must have exactly {self.n_words_per_example} words per example"

        return BinaryClassificationExample(value=" ".join(sampled), label=False)


#################
# DISCIMINATORS #
#################


class NegationRuleDiscriminationBehaviour(BinaryClassificationDiscriminationBehaviour):
    def __init__(self, rule: Rule):
        assert isinstance(rule, Rule), "Rule must be of type `Rule`"
        self.rule = rule

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        return not self.rule(example)


class BinaryOperatorDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(self, rules: Tuple[Rule], operator: Callable[[List[bool]], bool]):
        assert (
            len(rules) >= 1
        ), "Must have at least one rule to generate conjunction rule"
        assert all(
            isinstance(r, Rule) for r in rules
        ), "All rules must be of type `Rule`"

        self.rules = rules
        self.operator = operator

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        return self.operator([r(example) for r in self.rules])


class ConjunctionRuleDiscriminationBehaviour(BinaryOperatorDiscriminationBehaviour):
    def __init__(self, rules: Tuple[Rule]):
        super().__init__(rules, all)


class DisjunctionRuleDiscriminationBehaviour(BinaryOperatorDiscriminationBehaviour):
    def __init__(self, rules: Tuple[Rule]):
        super().__init__(rules, any)


class XorRuleDiscriminationBehaviour(BinaryOperatorDiscriminationBehaviour):
    def __init__(self, rules: Tuple[Rule]):
        super().__init__(rules, lambda x: sum(x) == 1)


# %%


class ContainsSpaceSeparatedFeatureDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(
        self,
        word: str,
        min_count: int,
        max_count: int,
        allowed_feature_idxs: Optional[Sequence[int]] = None,
        feature_must_appear_idxs: Optional[Sequence[int]] = None,
    ):
        self.word = word
        self.min_count = min_count
        self.max_count = max_count
        self.allowed_feature_idxs = allowed_feature_idxs
        self.feature_must_appear_idxs = feature_must_appear_idxs

        if self.feature_must_appear_idxs is None:
            self.feature_must_appear_idxs = ()

        assert len(word) >= 1, "Word cannot be the empty string"
        assert (
            1 <= self.min_count <= self.max_count
        ), "Word must appear at least once and cannot be bigger than `max_count`"

        if self.allowed_feature_idxs is not None:
            assert self.min_count <= len(
                self.allowed_feature_idxs
            ), "Must have enough feature indices to satisfy `min_count` constraint"

        if (
            self.allowed_feature_idxs is not None
            and self.feature_must_appear_idxs is not None
        ):
            assert set(self.feature_must_appear_idxs).issubset(
                set(self.allowed_feature_idxs)
            ), (
                f"{self.feature_must_appear_idxs=} must be a subset of"
                f" {self.allowed_feature_idxs=}"
            )

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")
        n_words_per_example = len(words)

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`."
        assert all(
            0 <= i < n_words_per_example for i in self.feature_must_appear_idxs
        ), f"Feature can't appear in out of range indices. Got '{example.value}'."

        if self.allowed_feature_idxs is None:
            allowed_feature_idxs = range(n_words_per_example)

        if not self.min_count <= words.count(self.word) <= self.max_count:
            return False

        for i in range(n_words_per_example):
            if i in allowed_feature_idxs:
                continue

            if words[i] == self.word:
                return False

        for i in self.feature_must_appear_idxs:
            if words[i] != self.word:
                return False

        return True


# %%


class DoesNotContainSpaceSeparatedFeatureDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(self, word: str):
        assert len(word) >= 1, "Word cannot be the empty string"
        self.word = word

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`."

        return self.word not in words


# %%


class ContainsWordFollowedByAnotherWordDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(
        self,
        first_feature: str,
        second_feature: str,
    ):
        assert len(first_feature) >= 1, "First feature cannot be the empty string"
        assert len(second_feature) >= 1, "Second feature cannot be the empty string"

        self.first_feature = first_feature
        self.second_feature = second_feature

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 2
        ), f"Expected example to contain at least two words but got `{example}`"

        if self.first_feature not in words:
            return False

        if self.second_feature not in words:
            return False

        return words.index(self.first_feature) < words.index(self.second_feature)


# %%


class ContainsAnySpaceSeparatedFeatureDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(
        self,
        words: Sequence[str],
    ):
        self.words = words

        assert len(words) >= 1, "Word cannot be the empty string"
        assert all(len(w) > 0 for w in words), "Words cannot be the empty string"

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`"

        return any(word in words for word in self.words)


# %%


class ContainsAllSpaceSeparatedFeaturesDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(
        self,
        words: Sequence[str],
    ):
        self.words = words

        assert len(words) >= 1, "Word cannot be the empty string"
        assert all(len(w) > 0 for w in words), "Words cannot be the empty string"

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`"

        return all(word in words for word in self.words)


# %%


class ContainsAnySubstringDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(
        self,
        substrings: Sequence[str],
    ):
        assert (
            len(substrings) >= 1
        ), f"Expected at least one substring but got `{substrings}`."
        assert all(
            len(s) > 0 for s in substrings
        ), "Substring can't be the empty string."

        self.substrings = substrings

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        # pre-conditions
        assert isinstance(example, BinaryClassificationExample), (
            "Expected example to be of type `BinaryClassificationExample` but got"
            f" `{type(example)}`."
        )
        assert isinstance(example.value, str), (
            "Expected example value to be of type `str` but got"
            f" `{type(example.value)}`."
        )

        # body
        out = any(substring in example.value for substring in self.substrings)

        # post-conditions
        assert isinstance(
            out, bool
        ), f"Expected return value to be of type `bool` but got `{type(out)}`."

        return out


# %%


class StartsWithWordDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(self, word: str):
        self.word = word
        assert len(word) >= 1, "Word cannot be the empty string"

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`"

        return words[0] == self.word


# %%


class EndsWithWordDiscriminationBehaviour(BinaryClassificationDiscriminationBehaviour):
    def __init__(self, word: str):
        self.word = word
        assert len(word) >= 1, "Word cannot be the empty string"

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`"

        return words[-1] == self.word


# %%


class WordsAreSortedAlphabeticallyDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(self, is_descending: bool):
        self.is_descending = is_descending

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`"

        return words == sorted(words, reverse=self.is_descending)


# %%


class ContainsAllSpaceSeparatedFeaturesDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(
        self,
        words: Sequence[str],
    ):
        self.words = words

        assert len(words) >= 1, "Word cannot be the empty string"
        assert all(len(w) > 0 for w in words), "Words cannot be the empty string"

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`"

        return all(word in words for word in self.words)


# %%


class WordsAreOrderedDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(self, features: Sequence[str], is_descending: bool):
        assert isinstance(is_descending, bool), "`is_descending` must be a bool"
        assert len(features) >= 2, "Must have two words for them to be ordered"
        assert all(len(w) > 0 for w in features), "Words cannot be the empty string"

        self.features = features
        self.is_descending = is_descending

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 2
        ), f"Expected example to contain at least two words but got `{example}`"

        if not all(feature in words for feature in self.features):
            return False

        word_indexes = [
            words.index(f) for f in sorted(self.features, reverse=self.is_descending)
        ]

        return word_indexes == sorted(word_indexes)


# %%


class StringIsCertainCaseDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(self, upper_case_is_passing: bool):
        self.upper_case_is_passing = upper_case_is_passing

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        assert (
            len(example.value) >= 1
        ), f"Expected example to contain at least one character but got `{example}`"
        assert (
            example.value.isupper() or example.value.islower()
        ), f"Expected example to be all upper or lower case but got `{example}`"

        if self.upper_case_is_passing:
            return example.value.isupper()

        return example.value.islower()


# %%


class StringEndsWithSubstringDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(self, substr: str, ends_with_is_passing: bool):
        self.substr = substr
        self.ends_with_is_passing = ends_with_is_passing

        assert len(substr) >= 1, "Substring cannot be the empty string"

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        assert (
            len(example.value) >= 1
        ), f"Expected example to contain at least one character but got `{example}`"

        if self.ends_with_is_passing:
            return example.value.endswith(self.substr)

        return not example.value.endswith(self.substr)


# %%


class ConjunctionOfDisjunctionDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(
        self,
        conjunction_of_disjunction_of_features: Sequence[Sequence[str]],
    ):
        assert (
            len(conjunction_of_disjunction_of_features) >= 1
        ), "Must have at least one disjunction of features"
        assert all(
            len(fs) >= 1 for fs in conjunction_of_disjunction_of_features
        ), "Must have at least one feature in each disjunction"
        assert all(
            all(map(lambda f: len(f) > 0, fs))
            for fs in conjunction_of_disjunction_of_features
        ), "Feature can't be the empty string"

        self.cd_features = conjunction_of_disjunction_of_features

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        assert (
            len(words) >= 1
        ), f"Expected example to contain at least one word but got `{example}`"

        for fs in self.cd_features:
            if not any(w in fs for w in words):
                return False

        return True


# %%


class SampledWordsAreWithinXPositionsOfEachOtherDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(self, x: int, w: str, w_prime: str):
        self.x = x
        self.w = w
        self.w_prime = w_prime

        assert len(w) > 0, "Word `w` cannot be the empty string"
        assert len(w_prime) > 0, "Word `w_prime` cannot be the empty string"
        assert x >= 1, "Number of positions must be at least 1"

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        words = example.value.split(" ")

        if self.w not in words:
            return False

        if self.w_prime not in words:
            return False

        w_index = words.index(self.w)
        w_prime_index = words.index(self.w_prime)

        return abs(w_index - w_prime_index) <= self.x


# %%


class ContainsSubstringDiscriminationBehaviour(
    BinaryClassificationDiscriminationBehaviour
):
    def __init__(
        self,
        substring: str,
        min_count: int,
        max_count: int,
    ):
        self.s = substring
        self.min_count = min_count
        self.max_count = max_count

        assert len(self.s) >= 1, "Substring cannot be the empty string"
        assert (
            1 <= self.min_count <= self.max_count
        ), "Substring must appear at least once and cannot be bigger than `max_count`"

    def is_passing_example(self, example: BinaryClassificationExample) -> bool:
        # pre-conditions
        assert isinstance(
            example.value, str
        ), f"Expected example's value to be a string."
        assert example.value != "", f"Expected example's value to be non-empty."

        # body
        if not self.min_count <= example.value.count(self.s) <= self.max_count:
            return False

        return True


###############
# DEPRECIATED #
###############


def generate_train_examples(
    r,
    r_prime,
    n_passing_examples: int,
    n_failing_examples: int,
    max_attempts: int,
) -> Tuple[BinaryClassificationExample]:
    """Generates examples which pass for both R and R' and fail for both R and R'."""

    passing_examples = []
    failing_examples = []

    assert r != r_prime, "R and R' must be different rules"
    assert max_attempts >= 1, "Must have at least one attempt per example to generate"
    assert n_passing_examples >= 0, "Must have non-negative number of passing examples"
    assert n_failing_examples >= 0, "Must have non-negative number of failing examples"

    for _ in range(n_passing_examples):
        for _ in range(max_attempts):
            words_r = r.generate_passing_example().value.split(" ")
            words_r_prime = r_prime.generate_passing_example().value.split(" ")

            if words_r is None or words_r_prime is None:
                continue  # try again

            assert len(words_r) == len(
                words_r_prime
            ), "R and R' must generate examples with the same number of words"

            n_words_per_example = len(words_r)

            words = words_r + words_r_prime
            words = random.sample(words, k=len(words))

            for candidate_words in itertools.permutations(words, r=n_words_per_example):
                candidate_example = BinaryClassificationExample(
                    value=" ".join(candidate_words),
                    label=True,
                )

                if r(candidate_example) and r_prime(candidate_example):
                    passing_examples.append(candidate_example)
                    break  # generated a passing example
            else:
                continue  # didn't break out of loop

            break  # broke out of loop

    assert len(passing_examples) == n_passing_examples, (
        f"Ran out of attempts at generating {n_passing_examples} passing examples;"
        f" generated {len(passing_examples)}"
    )
    assert all(
        r(example) and r_prime(example) for example in passing_examples
    ), "Expected all generated passing examples to pass for both R and R'"
    assert all(
        example.label for example in passing_examples
    ), "Expected all generated passing examples to have label True"

    for _ in range(n_failing_examples):
        for _ in range(max_attempts):
            words_r = r.generate_failing_example().value.split(" ")
            words_r_prime = r_prime.generate_failing_example().value.split(" ")

            assert len(words_r) == len(
                words_r_prime
            ), "R and R' must generate examples with the same number of words"

            n_words_per_example = len(words_r)

            words = words_r + words_r_prime
            words = random.sample(words, k=len(words))

            for candidate_words in itertools.permutations(words, r=n_words_per_example):
                candidate_example = BinaryClassificationExample(
                    value=" ".join(candidate_words),
                    label=False,
                )

                if (not r(candidate_example)) and (not r_prime(candidate_example)):
                    failing_examples.append(candidate_example)
                    break  # generated a failing example
            else:
                continue  # didn't break out of loop

            break  # broke out of loop

    assert len(failing_examples) == n_failing_examples, (
        f"Ran out of attempts at generating {n_failing_examples} failing examples;"
        f" generated {len(examples) - n_passing_examples}"
    )
    assert all(
        not r(example) and not r_prime(example) for example in failing_examples
    ), "Expected all generated failing examples to fail for both R and R'"
    assert all(
        not example.label for example in failing_examples
    ), "Expected all generated failing examples to have label False"

    examples = tuple(passing_examples + failing_examples)

    assert len(examples) == n_passing_examples + n_failing_examples, (
        f"Expected {n_passing_examples + n_failing_examples} examples; got"
        f" {len(examples)}"
    )

    return tuple(examples)


def generate_test_examples(
    r: Rule,
    r_prime: Rule,
    n_passing_examples: int,
    n_failing_examples: int,
    max_attempts: int,
) -> Tuple[BinaryClassificationExample]:
    """Generates examples which pass for R but not R' and fail for both R and R'."""

    passing_examples = []
    failing_examples = []

    assert r != r_prime, "R and R' must be different rules"
    assert max_attempts >= 1, "Must have at least one attempt per example to generate"
    assert n_passing_examples >= 0, "Must have non-negative number of passing examples"
    assert n_failing_examples >= 0, "Must have non-negative number of failing examples"

    for _ in range(n_passing_examples):
        for _ in range(max_attempts):
            words_r = r.generate_passing_example().value.split(" ")
            words_r_prime = r_prime.generate_failing_example().value.split(" ")

            assert len(words_r) == len(
                words_r_prime
            ), "R and R' must generate examples with the same number of words"

            for candidate_words in itertools.permutations(
                words_r + words_r_prime, r=len(words_r)
            ):
                candidate_example = BinaryClassificationExample(
                    value=" ".join(candidate_words),
                    label=True,
                )

                if r(candidate_example) and (not r_prime(candidate_example)):
                    passing_examples.append(candidate_example)
                    break  # generated a example which passes for R but not R'
            else:
                continue  # didn't break out of loop

            break  # broke out of loop

    assert len(passing_examples) == n_passing_examples, (
        f"Ran out of attempts at generating {n_passing_examples} passing examples;"
        f" generated {len(passing_examples)}"
    )
    assert all(
        r(example) and (not r_prime(example)) for example in passing_examples
    ), "Expected generated examples to pass for R but not R'"
    assert all(
        example.label for example in passing_examples
    ), "Expected all generated passing examples to have label True"

    for _ in range(n_failing_examples):
        for _ in range(max_attempts):
            words_r = r.generate_failing_example().value.split(" ")
            words_r_prime = r_prime.generate_failing_example().value.split(" ")

            assert len(words_r) == len(
                words_r_prime
            ), "R and R' must generate examples with the same number of words"

            for candidate_words in itertools.permutations(
                words_r + words_r_prime, r=len(words_r)
            ):
                candidate_example = BinaryClassificationExample(
                    value=" ".join(candidate_words),
                    label=False,
                )

                if (not r(candidate_example)) and (not r_prime(candidate_example)):
                    failing_examples.append(candidate_example)
                    break  # generated a failing example
            else:
                continue  # didn't break out of loop

            break  # broke out of loop

    assert len(failing_examples) == n_failing_examples, (
        f"Ran out of attempts at generating {n_failing_examples} failing examples;"
        f" generated {len(failing_examples)}"
    )
    assert all(
        (not r(example)) and (not r_prime(example)) for example in failing_examples
    ), "Expected generated examples to fail for both R and R'"
    assert all(
        not example.label for example in failing_examples
    ), "Expected all generated failing examples to have label False"

    examples = tuple(passing_examples + failing_examples)

    assert len(examples) == n_passing_examples + n_failing_examples, (
        f"Expected {n_passing_examples + n_failing_examples} examples; got"
        f" {len(examples)}"
    )

    return examples
