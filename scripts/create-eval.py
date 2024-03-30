import logging
import random
import json
import rules
import attacks
import datasets
import utils
import tiktoken

from rules import DEFAULT_RULE_FNS
from attacks import DEFAULT_ATTACK_BEHAVIOUR_CLASSES
from tqdm import tqdm
from pathlib import Path
from typing import Optional

config = {
    "seed": 0,
    "n_words_per_example": 5,
    "n_problems_per_rule_fn": 1,
    "logging_level": logging.INFO,
    "temperature": 0.0,
    "max_tries": 128,
}

rule_fns = list(DEFAULT_RULE_FNS)
features = datasets.get_common_words_features_dataset(
    shuffled=True, seed=config["seed"]
)
french_words = datasets.get_french_words()
tokenizer = tiktoken.encoding_for_model("davinci")

logging.basicConfig(format="%(asctime)s %(message)s", level=config["logging_level"])
logger = logging.getLogger(__name__)

N_SAMPLES = 32
N_FEW_SHOT_EXAMPLES = 64

ATTACK_BLACKLIST = {
    rules.contains_a_digit: {
        attacks.ChangeCaseAttackBehaviour,
        attacks.ChangeLanguageOfWordAttackBehaviour,
        attacks.ChangePositionOfWordAttackBehaviour,
        attacks.CommonMisspellingsOfWordAttackBehaviour,
        attacks.DifferentPartOfSpeechAttackBehaviour,
        attacks.InsertSpacesAttackBehaviour,
        attacks.JumbleLettersAttackBehaviour,
        attacks.MarkdownStylingAttackBehaviour,
        attacks.SynonymAttackBehaviour,
    },
    rules.contains_a_french_word: {
        attacks.ChangeCaseAttackBehaviour,
        attacks.ChangeLanguageOfWordAttackBehaviour,
        attacks.ChangePositionOfWordAttackBehaviour,
        attacks.CommonMisspellingsOfWordAttackBehaviour,
        attacks.DifferentPartOfSpeechAttackBehaviour,
        attacks.InsertSpacesAttackBehaviour,
        attacks.JumbleLettersAttackBehaviour,
        attacks.MarkdownStylingAttackBehaviour,
        attacks.SynonymAttackBehaviour,
    },
    rules.words_are_in_alphabetical_order: {
        attacks.ChangeCaseAttackBehaviour,
        attacks.ChangeLanguageOfWordAttackBehaviour,
        attacks.ChangePositionOfWordAttackBehaviour,
        attacks.CommonMisspellingsOfWordAttackBehaviour,
        attacks.DifferentPartOfSpeechAttackBehaviour,
        attacks.InsertSpacesAttackBehaviour,
        attacks.JumbleLettersAttackBehaviour,
        attacks.MarkdownStylingAttackBehaviour,
        attacks.SynonymAttackBehaviour,
    },
    rules.words_are_in_reverse_alphabetical_order: {
        attacks.ChangeCaseAttackBehaviour,
        attacks.ChangeLanguageOfWordAttackBehaviour,
        attacks.ChangePositionOfWordAttackBehaviour,
        attacks.CommonMisspellingsOfWordAttackBehaviour,
        attacks.DifferentPartOfSpeechAttackBehaviour,
        attacks.InsertSpacesAttackBehaviour,
        attacks.JumbleLettersAttackBehaviour,
        attacks.MarkdownStylingAttackBehaviour,
        attacks.SynonymAttackBehaviour,
    },
    rules.does_not_contain_the_word_w: {
        attacks.ChangeCaseAttackBehaviour,
        attacks.ChangeLanguageOfWordAttackBehaviour,
        attacks.ChangePositionOfWordAttackBehaviour,
        attacks.CommonMisspellingsOfWordAttackBehaviour,
        attacks.DifferentPartOfSpeechAttackBehaviour,
        attacks.InsertSpacesAttackBehaviour,
        attacks.JumbleLettersAttackBehaviour,
        attacks.MarkdownStylingAttackBehaviour,
        attacks.SynonymAttackBehaviour,
    },
    rules.ends_with_the_word_w: {
        attacks.DecreaseNumberOfWordsAttackBehaviour,  # TODO: Fix this; it's a bit of a hack
        attacks.RemoveSpacesAttackBehaviour,  # TODO: Fix this; it's a bit of a hack
    },
    rules.ends_with_the_word_w_or_contains_the_word_w_prime: {
        attacks.DecreaseNumberOfWordsAttackBehaviour,  # TODO: Fix this; it's a bit of a hack
        attacks.RemoveSpacesAttackBehaviour,  # TODO: Fix this; it's a bit of a hack
    },
}


def generate_binary_classification_task_with_attacks(
    rule_fn,
    rule: datasets.Rule,
    n_passing_examples: int,
    n_failing_examples: int,
    n_train_attacks: int,
    train_attack_behaviours: set[attacks.AdversarialAttackBehaviour],
    test_attack_behaviour: Optional[attacks.AdversarialAttackBehaviour],
    instruction: Optional[str] = datasets.DEFAULT_PROMPT_INSTRUCTION,
) -> datasets.BinaryClassificationTask:
    """Generates a `BinaryClassificationTask` for the rule with adversarial attacks in prompt. Last example is attacked with `test_attack_behaviour`."""

    # pre-conditions
    blacklist = ATTACK_BLACKLIST.get(rule_fn, set())

    if n_train_attacks > 0:
        assert (
            len(train_attack_behaviours) > 0
        ), "Must have train attack behaviours if `n_train_attacks` > 0"

    for attack in train_attack_behaviours:
        assert (
            attack not in blacklist
        ), f"Cannot attack {rule.__name__} with {attack.__name__}"

    if test_attack_behaviour is not None:
        assert (
            test_attack_behaviour not in blacklist
        ), f"Cannot attack {rule.__name__} with {test_attack_behaviour.__name__}"

    # body
    task = None

    for _ in range(config["max_tries"]):
        try:
            task = rule.generate_binary_classification_task(
                instruction=instruction,
                n_passing_examples=n_passing_examples,
                n_failing_examples=n_failing_examples,
                ends_with="passing",
            )
        except AssertionError:
            continue  # try again
        else:
            break  # success

    if task is None:
        raise ValueError(
            f"Could not generate binary classification task for '{rule.explanation}'"
        )

    train_attacked_examples = []
    passing_examples = [rule.generate_passing_example() for _ in range(n_train_attacks)]
    shuffled_attack_behaviours = utils.sample(
        train_attack_behaviours,
        k=len(train_attack_behaviours),
    )

    for i, example in enumerate(passing_examples):
        idx = i % len(shuffled_attack_behaviours)
        attack_behaviour = shuffled_attack_behaviours[idx]
        attack = attacks.AdversarialAttack(rule, attack_behaviour)

        attacked_example = None

        for _ in range(config["max_tries"]):
            example = rule.generate_passing_example()

            try:
                attacked_example = attack.generate_adversarial_passing_example(example)
            except AssertionError as e:
                logging.info(
                    f"Failed to generate adversarial example for '{rule.explanation}'"
                    f" with input '{example.value}' and attack"
                    f" '{attack_behaviour}'. {e}"
                )
                continue  # try again
            else:
                break  # success

        if attacked_example is None:
            raise ValueError(
                f"Could not generate adversarial example for '{rule.explanation}' with"
                f" input '{example.value}' and attack '{attack_behaviour}'"
            )

        train_attacked_examples.append(attacked_example)

    train_examples = tuple(task.examples[:-1]) + tuple(train_attacked_examples)
    train_examples = utils.sample(train_examples, k=len(train_examples))

    test_example = (
        rule.generate_passing_example()
        if random.random() < 0.5
        else rule.generate_failing_example()
    )

    if test_attack_behaviour:
        attack = attacks.AdversarialAttack(rule, test_attack_behaviour)

        test_example = None

        for _ in range(config["max_tries"]):
            try:
                example = rule.generate_passing_example()
                test_example = attack.generate_adversarial_passing_example(example)
            except AssertionError:
                continue  # try again
            else:
                break  # success

        if test_example is None:
            raise ValueError(
                f"Could not generate adversarial example for '{rule.explanation}' with"
                f" input '{example.value}' and attack '{attack_behaviour}'"
            )

    new_examples = train_examples + (test_example,)

    new_task = datasets.BinaryClassificationTask(
        instruction=task.instruction,
        examples=new_examples,
    )

    # post-conditions
    assert isinstance(new_task, datasets.BinaryClassificationTask)
    assert all(
        isinstance(example, datasets.BinaryClassificationExample)
        for example in new_task.examples
    )
    assert (
        len(new_task.examples)
        == n_passing_examples + n_failing_examples + n_train_attacks
    )
    assert new_task.instruction == task.instruction

    return new_task


def generate_multiple_choice_articulation_task_with_attacks(
    rule_fn,
    rule: datasets.Rule,
    other_rule: datasets.Rule,
    n_passing_examples: int,
    n_failing_examples: int,
    n_train_attacks: int,
    train_attack_behaviours: set[attacks.AdversarialAttackBehaviour],
    instruction: Optional[str] = datasets.DEFAULT_PROMPT_INSTRUCTION,
) -> datasets.MultipleChoiceExplanationTask:
    """Generates a `MultipleChoiceExplanationTask` for the rule with adversarial attacks in prompt."""

    # pre-conditions
    blacklist = ATTACK_BLACKLIST.get(rule_fn, set())

    if n_train_attacks > 0:
        assert (
            len(train_attack_behaviours) > 0
        ), "Must have train attack behaviours if `n_train_attacks` > 0"

    for attack in train_attack_behaviours:
        assert (
            attack not in blacklist
        ), f"Cannot attack {rule.__name__} with {attack.__name__}"

    # body
    task = None

    for _ in range(config["max_tries"]):
        try:
            task = rule.generate_multiple_choice_explanation_task(
                instruction=instruction,
                n_passing_examples=n_passing_examples,
                n_failing_examples=n_failing_examples,
                incorrect_choices=(other_rule.explanation,),
            )
        except AssertionError:
            continue  # try again
        else:
            break  # success

    if task is None:
        raise ValueError(
            f"Could not generate binary classification task for '{rule.explanation}'"
        )

    train_attacked_examples = []
    passing_examples = [rule.generate_passing_example() for _ in range(n_train_attacks)]
    shuffled_attack_behaviours = utils.sample(
        train_attack_behaviours,
        k=len(train_attack_behaviours),
    )

    for i, example in enumerate(passing_examples):
        idx = i % len(shuffled_attack_behaviours)
        attack_behaviour = shuffled_attack_behaviours[idx]
        attack = attacks.AdversarialAttack(rule, attack_behaviour)

        attacked_example = None

        for _ in range(config["max_tries"]):
            example = rule.generate_passing_example()

            try:
                attacked_example = attack.generate_adversarial_passing_example(example)
            except AssertionError as e:
                logging.info(
                    f"Failed to generate adversarial example for '{rule.explanation}'"
                    f" with input '{example.value}' and attack"
                    f" '{attack_behaviour}'. {e}"
                )
                continue  # try again
            else:
                break  # success

        if attacked_example is None:
            raise ValueError(
                f"Could not generate adversarial example for '{rule.explanation}' with"
                f" input '{example.value}' and attack '{attack_behaviour}'"
            )

        train_attacked_examples.append(attacked_example)

    train_examples = tuple(task.classification_task.examples) + tuple(
        train_attacked_examples
    )
    new_examples = utils.sample(train_examples, k=len(train_examples))

    new_classification_task = datasets.BinaryClassificationTask(
        instruction=None,
        examples=new_examples,
    )

    new_task = datasets.MultipleChoiceExplanationTask(
        instruction=task.instruction,
        classification_task=new_classification_task,
        choices=task.choices,
        answer=task.answer,
    )

    # post-conditions
    assert isinstance(new_task, datasets.MultipleChoiceExplanationTask)
    assert all(
        isinstance(example, datasets.BinaryClassificationExample)
        for example in new_task.classification_task.examples
    )
    assert (
        len(new_task.classification_task.examples)
        == n_passing_examples + n_failing_examples + n_train_attacks
    )
    assert new_task.instruction == task.instruction

    return new_task


# %%
def generate_freeform_articulation_task_with_attacks(
    rule_fn,
    rule: datasets.Rule,
    n_passing_examples: int,
    n_failing_examples: int,
    n_train_attacks: int,
    train_attack_behaviours: set[attacks.AdversarialAttackBehaviour],
    instruction: Optional[str] = datasets.DEFAULT_PROMPT_INSTRUCTION,
) -> datasets.FreeformExplanationTask:
    """Generates a `FreeformExplanationTask` for the rule with adversarial attacks in prompt."""

    # pre-conditions
    blacklist = ATTACK_BLACKLIST.get(rule_fn, set())

    if n_train_attacks > 0:
        assert (
            len(train_attack_behaviours) > 0
        ), "Must have train attack behaviours if `n_train_attacks` > 0"

    for attack in train_attack_behaviours:
        assert (
            attack not in blacklist
        ), f"Cannot attack {rule.__name__} with {attack.__name__}"

    # body
    task = None

    for _ in range(config["max_tries"]):
        try:
            task = rule.generate_freeform_explanation_task(
                instruction=instruction,
                n_passing_examples=n_passing_examples,
                n_failing_examples=n_failing_examples,
            )
        except AssertionError:
            continue  # try again
        else:
            break  # success

    if task is None:
        raise ValueError(
            f"Could not generate binary classification task for '{rule.explanation}'"
        )

    train_attacked_examples = []
    passing_examples = [rule.generate_passing_example() for _ in range(n_train_attacks)]
    shuffled_attack_behaviours = utils.sample(
        train_attack_behaviours,
        k=len(train_attack_behaviours),
    )

    for i, example in enumerate(passing_examples):
        idx = i % len(shuffled_attack_behaviours)
        attack_behaviour = shuffled_attack_behaviours[idx]
        attack = attacks.AdversarialAttack(rule, attack_behaviour)

        attacked_example = None

        for _ in range(config["max_tries"]):
            example = rule.generate_passing_example()

            try:
                attacked_example = attack.generate_adversarial_passing_example(example)
            except AssertionError as e:
                logging.info(
                    f"Failed to generate adversarial example for '{rule.explanation}'"
                    f" with input '{example.value}' and attack"
                    f" '{attack_behaviour}'. {e}"
                )
                continue  # try again
            else:
                break  # success

        if attacked_example is None:
            raise ValueError(
                f"Could not generate adversarial example for '{rule.explanation}' with"
                f" input '{example.value}' and attack '{attack_behaviour}'"
            )

        train_attacked_examples.append(attacked_example)

    train_examples = tuple(task.classification_task.examples) + tuple(
        train_attacked_examples
    )
    new_examples = utils.sample(train_examples, k=len(train_examples))

    new_classification_task = datasets.BinaryClassificationTask(
        instruction=None,
        examples=new_examples,
    )

    new_task = datasets.FreeformExplanationTask(
        instruction=task.instruction,
        classification_task=new_classification_task,
        explanation=task.explanation,
    )

    # post-conditions
    assert isinstance(new_task, datasets.FreeformExplanationTask)
    assert all(
        isinstance(example, datasets.BinaryClassificationExample)
        for example in new_task.classification_task.examples
    )
    assert (
        len(new_task.classification_task.examples)
        == n_passing_examples + n_failing_examples + n_train_attacks
    )
    assert new_task.instruction == task.instruction

    return new_task


# %%
def get_number_of_lines(fname: Path) -> int:
    """Return the number of lines in the file."""
    if not fname.exists():
        return 0
    with open(fname, "r") as f:
        return sum(1 for _ in f)


# %% [markdown]
# # Classification

# %% [markdown]
# ### Fine-tuning in-distribution classification


# %%
def generate_classification_fine_tuning_in_distribution_problems(
    n: int, k: int
) -> list[dict]:
    assert n >= 1, "need at least 1 problem"
    assert k >= 2, "need at least 2 few-shot examples"
    assert k % 2 == 0, "need an even number of few-shot examples"

    records = []

    n_passing_examples = k // 2
    n_failing_examples = k // 2

    for seed in tqdm(range(n)):
        # use same features for each rule function
        random.seed(seed)
        w = random.choice(features)
        (w_prime,) = utils.sample(features, exclude={w}, k=1)
        feature_idx = random.choice(range(config["n_words_per_example"]))

        for rule_fn in rule_fns:
            # random.seed(seed)
            rule = rule_fn(
                w=w,
                w_prime=w_prime,
                features=features,
                k=3,
                k_prime=5,
                i=feature_idx,
                x=1,
                french_words=french_words,
                n_words_per_example=config["n_words_per_example"],
            )

            logger.debug(f"Generating task for rule `{rule_fn.__name__}`")

            task = rule.generate_binary_classification_task(
                n_passing_examples=n_passing_examples,
                n_failing_examples=n_failing_examples,
            )

            logger.debug(f"Finished.")

            problem = datasets.binary_classification_task_to_problem(task)

            records.append(
                {
                    "rule_fn": rule_fn,
                    "rule": rule,
                    "prompt": problem.prompt,
                    "expected_completion": problem.expected_completion,
                }
            )

    return records


# len(generate_classification_fine_tuning_in_distribution_problems(n=1, k=32))

# %%
if False:
    path = Path("data/classification/fine-tuning/in-distribution/")
    records = generate_classification_fine_tuning_in_distribution_problems(
        n=N_SAMPLES, k=N_FEW_SHOT_EXAMPLES
    )

    # clear all files that correspond to each record
    for record in records:
        fname = path / f"{record['rule_fn'].__name__}.jsonl"
        with open(fname, "w") as f:
            pass

    # write jsonl files
    for record in records:
        with open(path / f"{record['rule_fn'].__name__}.jsonl", "a") as f:
            data = {"input": record["prompt"], "ideal": record["expected_completion"]}
            line = json.dumps(data)
            f.write(line + "\n")

# %% [markdown]
# ### Fine-tuning out-of-distribution classification


# %%
def generate_classification_fine_tuning_out_of_distribution_problems(
    n: int, k: int
) -> list[dict]:
    assert n >= 1, "need at least 1 problem"
    assert k >= 2, "need at least 2 few-shot examples"
    assert k % 2 == 0, "need an even number of few-shot examples"

    records = []

    n_passing_examples = k // 2
    n_failing_examples = k // 2

    for seed in tqdm(range(n)):
        # use same features for each rule function
        random.seed(seed)
        w = random.choice(features)
        (w_prime,) = utils.sample(features, exclude={w}, k=1)
        feature_idx = random.choice(range(config["n_words_per_example"]))

        for rule_fn in rule_fns:
            attack_blacklist = ATTACK_BLACKLIST.get(rule_fn, set())
            attack_behaviour_classes = set(
                attack
                for attack in DEFAULT_ATTACK_BEHAVIOUR_CLASSES
                if attack not in attack_blacklist
            )

            for attack_behaviour_class in attack_behaviour_classes:
                # random.seed(seed)
                rule = rule_fn(
                    w=w,
                    w_prime=w_prime,
                    features=features,
                    k=3,
                    k_prime=5,
                    i=feature_idx,
                    x=1,
                    french_words=french_words,
                    n_words_per_example=config["n_words_per_example"],
                )

                logger.debug(f"Generating task for rule `{rule_fn.__name__}`")

                try:
                    attack_behaviour = attack_behaviour_class(
                        rule=rule, w=w, features=features, max_words=5, seed=seed
                    )
                except AssertionError as e:
                    logging.warning(f"Skipping rule '{rule.explanation}'. {e}")
                    continue

                attack = attacks.AdversarialAttack(rule, attack_behaviour)
                task = None

                for _ in range(config["max_tries"]):
                    try:
                        task = rule.generate_binary_classification_task(
                            n_passing_examples=n_passing_examples,
                            n_failing_examples=n_failing_examples,
                            ends_with="passing",
                        )
                    except AssertionError as e:
                        continue
                    else:
                        break  # successfully generated task; stop trying

                if task is None:
                    logging.warning(
                        f"Skipping rule '{rule.explanation}'. Could not generate task."
                    )
                    continue

                try:
                    adversarial_task = attack.generate_adversarial_task(task)
                except AssertionError as e:
                    logging.warning(
                        f"Skipping {attack_behaviour_class.__name__} for rule"
                        f" '{rule.explanation}'. {e}"
                    )
                    continue

                logger.debug(f"Finished.")
                logger.debug(
                    f"Creating adversarial problem for rule `{rule_fn.__name__}` and"
                    f" attack `{attack_behaviour_class.__name__}`"
                )

                adversarial_problem = datasets.binary_classification_task_to_problem(
                    adversarial_task
                )

                logger.debug(f"Finished.")

                records.append(
                    {
                        "attack_behaviour_class": attack_behaviour_class,
                        "rule_fn": rule_fn,
                        "prompt": adversarial_problem.prompt,
                        "expected_completion": adversarial_problem.expected_completion,
                    }
                )

    return records


# len(generate_classification_fine_tuning_out_of_distribution_problems(n=1, k=32))

# %%
if False:
    path = Path("data/classification/fine-tuning/out-of-distribution/")
    records = generate_classification_fine_tuning_out_of_distribution_problems(
        n=N_SAMPLES, k=N_FEW_SHOT_EXAMPLES
    )

    # clear all files that correspond to each record
    for record in records:
        for train_attack_behaviour_class in DEFAULT_ATTACK_BEHAVIOUR_CLASSES:
            fname = (
                path
                / f"{record['rule_fn'].__name__}"
                / f"{train_attack_behaviour_class.__name__}.jsonl"
            )

            if not fname.parent.exists():
                fname.parent.mkdir(parents=True)

            with open(fname, "w") as f:
                pass

    # write jsonl files
    for record in records:
        fname = (
            path
            / f"{record['rule_fn'].__name__}"
            / f"{record['attack_behaviour_class'].__name__}.jsonl"
        )
        with open(fname, "a") as f:
            data = {"input": record["prompt"], "ideal": record["expected_completion"]}
            line = json.dumps(data)
            f.write(line + "\n")

# %% [markdown]
# ### In-context in-distribution


# %%
def generate_classification_in_context_in_distribution_problems(
    n: int, k: int, skip_rule_fns=set()
) -> list[dict]:
    assert n >= 1, "need at least 1 problem"
    assert k >= 2, "need at least 2 few-shot examples"
    assert k % 4 == 0, "need an number of few-shot examples to be divisible by 4"

    records = []

    n_train_attacks = k // 2
    n_passing_examples = k // 4
    n_failing_examples = k // 4

    for seed in tqdm(range(n)):
        random.seed(seed)
        w = random.choice(features)
        (w_prime,) = utils.sample(features, exclude={w}, k=1)
        feature_idx = random.choice(range(0, config["n_words_per_example"] - 1))

        for rule_fn in rule_fns:
            if rule_fn in skip_rule_fns:
                continue

            rule = rule_fn(
                w=w,
                w_prime=w_prime,
                features=features,
                k=3,
                k_prime=5,
                i=feature_idx,
                x=1,
                french_words=french_words,
                n_words_per_example=config["n_words_per_example"],
            )

            attack_blacklist = ATTACK_BLACKLIST.get(rule_fn, set())
            attack_behaviour_classes = set(
                attack
                for attack in DEFAULT_ATTACK_BEHAVIOUR_CLASSES
                if attack not in attack_blacklist
            )
            rule_features = utils.get_rule_features(rule)
            attack_behaviours = tuple()

            for attack_behaviour_class in attack_behaviour_classes:
                try:
                    attack_behaviour = attack_behaviour_class(
                        rule=rule,
                        w=rule_features,
                        features=features,
                        max_words=5,
                    )
                except AssertionError as e:
                    logging.error(
                        f"Failed to define {attack_behaviour_class.__name__} for rule"
                        f" '{rule.explanation}' with features {rule_features}."
                    )
                    raise AssertionError(
                        f"Failed to define {attack_behaviour_class.__name__} for rule"
                        f" '{rule.explanation}' with features {rule_features}."
                    )

                attack_behaviours += (attack_behaviour,)

            task = generate_binary_classification_task_with_attacks(
                rule_fn=rule_fn,
                rule=rule,
                n_passing_examples=n_passing_examples,
                n_failing_examples=n_failing_examples,
                n_train_attacks=n_train_attacks,
                train_attack_behaviours=attack_behaviours,
                test_attack_behaviour=None,  # final example is in-distribution
            )

            problem = datasets.binary_classification_task_to_problem(task)

            records.append(
                {
                    "rule_fn": rule_fn,
                    "prompt": problem.prompt,
                    "expected_completion": problem.expected_completion,
                }
            )

    return records


# len(generate_classification_in_context_in_distribution_problems(n=1, k=32))

# %%
path = Path("data/classification/in-context/in-distribution")

# determine the rule functions that we can skip
skip_rule_fns = set()

for rule_fn in DEFAULT_RULE_FNS:
    fname = path / f"{rule_fn.__name__}.jsonl"
    if get_number_of_lines(fname) >= N_SAMPLES:
        skip_rule_fns.add(rule_fn.__name__)

records = generate_classification_in_context_in_distribution_problems(
    n=N_SAMPLES, k=N_FEW_SHOT_EXAMPLES, skip_rule_fns=skip_rule_fns
)

# clear all files that correspond to each record
for record in records:
    fname = path / f"{record['rule_fn'].__name__}.jsonl"

    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    with open(fname, "w") as f:
        pass

# write jsonl files
for record in records:
    fname = path / f"{record['rule_fn'].__name__}.jsonl"
    with open(fname, "a") as f:
        data = {"input": record["prompt"], "ideal": record["expected_completion"]}
        line = json.dumps(data)
        f.write(line + "\n")

# %% [markdown]
# ### In-context out-of-distribution


# %%
def generate_classification_in_context_out_of_distribution_problems(
    n: int, k: int, skip_rule_fns=set()
) -> list[dict]:
    assert n >= 1, "need at least 1 problem"
    assert k >= 2, "need at least 2 few-shot examples"
    assert k % 4 == 0, "need an number of few-shot examples to be divisible by 4"

    records = []

    n_train_attacks = k // 2
    n_passing_examples = k // 4
    n_failing_examples = k // 4

    for seed in tqdm(range(n)):
        random.seed(seed)
        w = random.choice(features)
        (w_prime,) = utils.sample(features, exclude={w}, k=1)
        feature_idx = random.choice(range(0, config["n_words_per_example"] - 1))

        for (
            test_attack_behaviour_class,
            train_attack_behaviour_classes,
        ) in utils.k_folds(DEFAULT_ATTACK_BEHAVIOUR_CLASSES):
            for rule_fn in rule_fns:
                if rule_fn in skip_rule_fns:
                    continue

                attack_blacklist = ATTACK_BLACKLIST.get(rule_fn, set())

                if test_attack_behaviour_class in attack_blacklist:
                    continue  # skip this rule function

                rule = rule_fn(
                    w=w,
                    w_prime=w_prime,
                    features=features,
                    k=3,
                    k_prime=5,
                    i=feature_idx,
                    x=1,
                    french_words=french_words,
                    n_words_per_example=config["n_words_per_example"],
                )

                train_attack_behaviour_classes = set(
                    attack
                    for attack in DEFAULT_ATTACK_BEHAVIOUR_CLASSES
                    if attack not in attack_blacklist
                    and attack != test_attack_behaviour_class
                )
                rule_features = utils.get_rule_features(rule)
                attack_behaviours = tuple()

                assert test_attack_behaviour_class not in train_attack_behaviour_classes

                test_attack_behaviour = test_attack_behaviour_class(
                    rule=rule,
                    w=rule_features,
                    features=features,
                    max_words=5,
                )

                for train_attack_behaviour_class in train_attack_behaviour_classes:
                    try:
                        attack_behaviour = train_attack_behaviour_class(
                            rule=rule,
                            w=rule_features,
                            features=features,
                            max_words=5,
                        )
                    except AssertionError as e:
                        logging.error(
                            "Failed to define"
                            f" {train_attack_behaviour_class.__name__} for rule"
                            f" '{rule.explanation}' with features {rule_features}."
                        )
                        raise AssertionError(
                            "Failed to define"
                            f" {train_attack_behaviour_class.__name__} for rule"
                            f" '{rule.explanation}' with features {rule_features}."
                        )

                    attack_behaviours += (attack_behaviour,)

                task = generate_binary_classification_task_with_attacks(
                    rule_fn=rule_fn,
                    rule=rule,
                    n_passing_examples=n_passing_examples,
                    n_failing_examples=n_failing_examples,
                    n_train_attacks=n_train_attacks,
                    train_attack_behaviours=attack_behaviours,
                    test_attack_behaviour=test_attack_behaviour,  # final example is out-of-distribution
                )

                problem = datasets.binary_classification_task_to_problem(task)

                records.append(
                    {
                        "test_attack_behaviour_class": test_attack_behaviour_class,
                        "rule_fn": rule_fn,
                        "prompt": problem.prompt,
                        "expected_completion": problem.expected_completion,
                    }
                )

    return records


# len(generate_classification_in_context_out_of_distribution_problems(n=1, k=32))

# %%
path = Path("data/classification/in-context/out-of-distribution/")

# determine the rule functions that we can skip
skip_rule_fns = set()

for rule_fn in DEFAULT_RULE_FNS:
    for attack_behaviour_class in DEFAULT_ATTACK_BEHAVIOUR_CLASSES:
        fname = (
            path
            / f"{record['rule_fn'].__name__}"
            / f"{attack_behaviour_class.__name__}.jsonl"
        )
        if get_number_of_lines(fname) >= N_SAMPLES:
            skip_rule_fns.add(rule_fn.__name__)

records = generate_classification_in_context_out_of_distribution_problems(
    n=N_SAMPLES, k=N_FEW_SHOT_EXAMPLES, skip_rule_fns=skip_rule_fns
)

# clear all files that correspond to each record
for record in records:
    for attack_behaviour_class in DEFAULT_ATTACK_BEHAVIOUR_CLASSES:
        fname = (
            path
            / f"{record['rule_fn'].__name__}"
            / f"{attack_behaviour_class.__name__}.jsonl"
        )

        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)

        with open(fname, "w") as f:
            pass

# write jsonl files
for record in records:
    fname = (
        path
        / f"{record['rule_fn'].__name__}"
        / f"{record['test_attack_behaviour_class'].__name__}.jsonl"
    )
    with open(fname, "a") as f:
        data = {"input": record["prompt"], "ideal": record["expected_completion"]}
        line = json.dumps(data)
        f.write(line + "\n")

# %% [markdown]
# # Articulation

# %% [markdown]
# ### In-context multiple choice


# %%
def generate_mc_articulation_in_context_problems(
    n: int, k: int, skip_rule_fns=set()
) -> list[dict]:
    assert n >= 1, "need at least 1 problem"
    assert k >= 2, "need at least 2 few-shot examples"
    assert k % 4 == 0, "need an number of few-shot examples to be divisible by 4"

    records = []

    n_train_attacks = k // 2
    n_passing_examples = k // 4
    n_failing_examples = k // 4

    for seed in tqdm(range(n)):
        random.seed(seed)
        w = random.choice(features)
        (w_prime,) = utils.sample(features, exclude={w}, k=1)
        feature_idx = random.choice(range(0, config["n_words_per_example"] - 1))

        for rule_fn in rule_fns:
            if rule_fn in skip_rule_fns:
                continue

            attack_blacklist = ATTACK_BLACKLIST.get(rule_fn, set())

            rule = rule_fn(
                w=w,
                w_prime=w_prime,
                features=features,
                k=3,
                k_prime=5,
                i=feature_idx,
                x=1,
                french_words=french_words,
                n_words_per_example=config["n_words_per_example"],
            )

            other_rule_fn = None

            for _ in range(config["max_tries"]):
                (other_rule_fn,) = utils.sample(
                    DEFAULT_RULE_FNS, exclude={rule_fn}, k=1
                )
                other_rule = other_rule_fn(
                    w=w,
                    w_prime=w_prime,
                    features=features,
                    k=3,
                    k_prime=5,
                    i=feature_idx,
                    x=1,
                    french_words=french_words,
                    n_words_per_example=config["n_words_per_example"],
                )
                some_passing_examples = [
                    rule.generate_passing_example() for _ in range(4)
                ]
                some_failing_examples = [
                    rule.generate_failing_example() for _ in range(4)
                ]
                some_examples = some_passing_examples + some_failing_examples
                is_indistinguishable = all(
                    rule(example) == other_rule(example) for example in some_examples
                )

                if not is_indistinguishable:
                    break
            else:
                raise ValueError(
                    "Could not find a rule that is distinguishable from"
                    f" '{rule.explanation}'"
                )

            train_attack_behaviour_classes = set(
                attack
                for attack in DEFAULT_ATTACK_BEHAVIOUR_CLASSES
                if attack not in attack_blacklist
            )
            rule_features = utils.get_rule_features(rule)
            attack_behaviours = tuple()

            for train_attack_behaviour_class in train_attack_behaviour_classes:
                try:
                    attack_behaviour = train_attack_behaviour_class(
                        rule=rule,
                        w=rule_features,
                        features=features,
                        max_words=5,
                    )
                except AssertionError as e:
                    logging.error(
                        f"Failed to define {train_attack_behaviour_class.__name__} for"
                        f" rule '{rule.explanation}' with features {rule_features}."
                    )
                    raise AssertionError(
                        f"Failed to define {train_attack_behaviour_class.__name__} for"
                        f" rule '{rule.explanation}' with features {rule_features}."
                    )

                attack_behaviours += (attack_behaviour,)

            task = generate_multiple_choice_articulation_task_with_attacks(
                rule_fn=rule_fn,
                rule=rule,
                other_rule=other_rule,
                n_passing_examples=n_passing_examples,
                n_failing_examples=n_failing_examples,
                n_train_attacks=n_train_attacks,
                train_attack_behaviours=attack_behaviours,
            )

            problem = datasets.multiple_choice_explanation_task_to_problem(task)

            records.append(
                {
                    "rule_fn": rule_fn,
                    "prompt": problem.prompt,
                    "expected_completion": problem.expected_completion + ")",
                }
            )

    return records


# %%
path = Path("data/articulation/in-context/multiple-choice/")

# determine the rule functions that we can skip
skip_rule_fns = set()

for rule_fn in DEFAULT_RULE_FNS:
    fname = path / f"{rule_fn.__name__}.jsonl"
    if get_number_of_lines(fname) >= N_SAMPLES:
        skip_rule_fns.add(rule_fn.__name__)

records = generate_mc_articulation_in_context_problems(
    n=N_SAMPLES, k=N_FEW_SHOT_EXAMPLES, skip_rule_fns=skip_rule_fns
)

# clear all files that correspond to each record
for record in records:
    fname = path / f"{record['rule_fn'].__name__}.jsonl"

    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    with open(fname, "w") as f:
        pass

# write jsonl files
for record in records:
    fname = path / f"{record['rule_fn'].__name__}.jsonl"
    with open(fname, "a") as f:
        data = {"input": record["prompt"], "ideal": record["expected_completion"]}
        line = json.dumps(data)
        f.write(line + "\n")

# %% [markdown]
# ### In-context freeform articulation


# %%
def generate_freeform_articulation_in_context_problems(
    n: int, k: int, skip_rule_fns=set()
) -> list[dict]:
    assert n >= 1, "need at least 1 problem"
    assert k >= 2, "need at least 2 few-shot examples"
    assert k % 4 == 0, "need an number of few-shot examples to be divisible by 4"

    records = []

    n_train_attacks = k // 2
    n_passing_examples = k // 4
    n_failing_examples = k // 4

    for seed in tqdm(range(n)):
        random.seed(seed)
        w = random.choice(features)
        (w_prime,) = utils.sample(features, exclude={w}, k=1)
        feature_idx = random.choice(range(0, config["n_words_per_example"] - 1))

        for rule_fn in rule_fns:
            if rule_fn in skip_rule_fns:
                continue

            attack_blacklist = ATTACK_BLACKLIST.get(rule_fn, set())

            rule = rule_fn(
                w=w,
                w_prime=w_prime,
                features=features,
                k=3,
                k_prime=5,
                i=feature_idx,
                x=1,
                french_words=french_words,
                n_words_per_example=config["n_words_per_example"],
            )

            other_rule_fn = None

            for _ in range(config["max_tries"]):
                (other_rule_fn,) = utils.sample(
                    DEFAULT_RULE_FNS, exclude={rule_fn}, k=1
                )
                other_rule = other_rule_fn(
                    w=w,
                    w_prime=w_prime,
                    features=features,
                    k=3,
                    k_prime=5,
                    i=feature_idx,
                    x=1,
                    french_words=french_words,
                    n_words_per_example=config["n_words_per_example"],
                )
                some_passing_examples = [
                    rule.generate_passing_example() for _ in range(4)
                ]
                some_failing_examples = [
                    rule.generate_failing_example() for _ in range(4)
                ]
                some_examples = some_passing_examples + some_failing_examples
                is_indistinguishable = all(
                    rule(example) == other_rule(example) for example in some_examples
                )

                if not is_indistinguishable:
                    break
            else:
                raise ValueError(
                    "Could not find a rule that is distinguishable from"
                    f" '{rule.explanation}'"
                )

            train_attack_behaviour_classes = set(
                attack
                for attack in DEFAULT_ATTACK_BEHAVIOUR_CLASSES
                if attack not in attack_blacklist
            )
            rule_features = utils.get_rule_features(rule)
            attack_behaviours = tuple()

            for train_attack_behaviour_class in train_attack_behaviour_classes:
                try:
                    attack_behaviour = train_attack_behaviour_class(
                        rule=rule,
                        w=rule_features,
                        features=features,
                        max_words=5,
                    )
                except AssertionError as e:
                    logging.error(
                        f"Failed to define {train_attack_behaviour_class.__name__} for"
                        f" rule '{rule.explanation}' with features {rule_features}."
                    )
                    raise AssertionError(
                        f"Failed to define {train_attack_behaviour_class.__name__} for"
                        f" rule '{rule.explanation}' with features {rule_features}."
                    )

                attack_behaviours += (attack_behaviour,)

            task = generate_freeform_articulation_task_with_attacks(
                rule_fn=rule_fn,
                rule=rule,
                n_passing_examples=n_passing_examples,
                n_failing_examples=n_failing_examples,
                n_train_attacks=n_train_attacks,
                train_attack_behaviours=attack_behaviours,
            )

            problem = datasets.freeform_explanation_task_to_problem(task)

            records.append(
                {
                    "rule_fn": rule_fn,
                    "prompt": problem.prompt,
                    "expected_completion": problem.expected_completion,
                }
            )

    return records


# %%
path = Path("data/articulation/in-context/freeform/")

# determine the rule functions that we can skip
skip_rule_fns = set()

for rule_fn in DEFAULT_RULE_FNS:
    fname = path / f"{rule_fn.__name__}.jsonl"
    if get_number_of_lines(fname) >= N_SAMPLES:
        skip_rule_fns.add(rule_fn.__name__)

records = generate_freeform_articulation_in_context_problems(
    n=N_SAMPLES, k=N_FEW_SHOT_EXAMPLES, skip_rule_fns=skip_rule_fns
)

# clear all files that correspond to each record
for record in records:
    fname = path / f"{record['rule_fn'].__name__}.jsonl"

    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    with open(fname, "w") as f:
        pass

# write jsonl files
for record in records:
    fname = path / f"{record['rule_fn'].__name__}.jsonl"
    with open(fname, "a") as f:
        data = {"input": record["prompt"], "ideal": record["expected_completion"]}
        line = json.dumps(data)
        f.write(line + "\n")
