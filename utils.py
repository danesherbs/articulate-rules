import os
import random
import json
import pandas as pd
import openai
import openai.error
import backoff
import diskcache
import functools
import collections
import dotenv
import json
import time
import database
import hashlib

from typeguard import typechecked
from pathlib import Path
from typing import (
    Sequence,
    TypeVar,
    Collection,
    Union,
    Tuple,
    Set,
    Dict,
    Generator,
    Any,
    Iterable,
)


T = TypeVar("T")  # declare type variable

DEFAULT_CACHE_DIR = "cache/"
cache = diskcache.Cache(
    DEFAULT_CACHE_DIR, size_limit=2**31
)  # max size for git lfs is 2GB :'( https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage

DF_LABELED_ARTICULATIONS: pd.DataFrame = database.get_table_as_df(
    "labeled_articulations"
)
DF_LABELED_ARTICULATIONS["is_equivalent"] = DF_LABELED_ARTICULATIONS[
    "is_equivalent"
].astype(bool)


class TrieNode:
    def __init__(
        self, children: Union[dict, None] = None, is_word: bool = False
    ) -> None:
        self.children = children if children is not None else {}
        self.is_word = is_word

    def __str__(self, level: int = 0) -> str:
        s = ""

        for c, node in self.children.items():
            s += "| " * level + c

            if node.is_word:
                s += " *"

            s += "\n"
            s += node.__str__(level + 1)

        return s

    def insert(self, word: str) -> None:
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True


def find_leaves(node: TrieNode, prefix: str = "") -> Set[str]:
    leaves = []

    for c, child in node.children.items():
        if child.is_word and len(child.children) == 0:
            leaves.append(prefix + c)
        leaves += find_leaves(child, prefix + c)

    return set(leaves)


def in_google_colab() -> bool:
    """Checks if in Google colab notebook."""
    try:
        import google.colab  # type: ignore
    except:
        return False

    return True


def sample(
    seq: Collection[T],
    k: int = 1,
    include: Union[None, set[T]] = None,
    exclude: Union[None, set[T]] = None,
) -> Tuple[T, ...]:
    """Samples `k` items from `seq` without replacement, optionally including or excluding items."""

    # pre-conditions
    assert isinstance(seq, Collection)
    assert k >= 0, "Number of items to sample must be non-negative"
    assert k <= len(seq), (
        f"Cannot sample {k} items without replacement from a sequence with"
        f" {len(seq)} items"
    )

    if include is None:
        include = set()

    if exclude is None:
        exclude = set()

    count = collections.Counter(seq)
    filtered_exclude = set(item for item in exclude if item in seq)

    assert (
        len(include) <= k
    ), "Number of items to include can't exceed number of items to sample"
    assert (
        len(seq) - sum(count[v] for v in filtered_exclude) >= k
    ), "Not enough items to sample from sequence after excluding items"
    assert (
        len(include.intersection(exclude)) == 0
    ), "Sets `include` and `exclude` must be mutually exclusive"

    # sample
    filtered_seq = [item for item in seq if item not in include.union(exclude)]
    sampled = random.sample(filtered_seq, k - len(include)) + list(include)
    sampled = random.sample(sampled, k)

    # post-conditions
    assert len(sampled) == k, "Sampled items must be of length `k`"

    for item in include:
        assert item in sampled, "Sampled items must include all items in `include`"

    for item in exclude:
        assert (
            item not in sampled
        ), "Sampled items must not include any items in `exclude`"

    return tuple(sampled)


def to_batches(elements: Collection[T], batch_size: int) -> Iterable[Sequence[T]]:
    """Generates batches of elements from the given iterable collection."""
    return (elements[i : i + batch_size] for i in range(0, len(elements), batch_size))


def k_folds(seq: Tuple[T, ...]) -> Generator[Tuple[T, Tuple[T, ...]], None, None]:
    """
    Yields all possible folds of a sequence of length `k`.
    """
    assert isinstance(seq, tuple)

    for k in range(len(seq)):
        yield seq[k], seq[:k] + seq[k + 1 :]


def append_df_to_file(df_to_append: pd.DataFrame, fname: str) -> pd.DataFrame:
    """Appends dataframe `df_to_append` to daatframe at `fname`."""
    if not os.path.exists(fname):
        pd.DataFrame([], columns=df_to_append.columns).to_csv(fname, index=False)

    df = pd.read_csv(fname)

    assert all(
        df.columns == df_to_append.columns
    ), f"Columns of df {df.columns} and df_to_append {df_to_append.columns} must match"

    pd.concat([df, df_to_append]).drop_duplicates().to_csv(fname, index=False)

    return pd.read_csv(fname)


@cache.memoize()
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def openai_completion_create(cache_non_deterministic_ok=False, **kwargs) -> Any:
    if os.getenv("OPENAI_API_KEY") is None:
        dotenv.load_dotenv()

    assert (
        os.getenv("OPENAI_API_KEY") is not None
    ), "OpenAI API key must be set in `.env`."

    if (
        not cache_non_deterministic_ok
        and "temperature" in kwargs
        and kwargs["temperature"] > 0
    ):
        raise ValueError("Responses for temperature > 0 shouldn't be cached.")

    return openai.Completion.create(**kwargs)


@cache.memoize()
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def openai_chatcompletion_create(cache_non_deterministic_ok=False, **kwargs) -> Any:
    if os.getenv("OPENAI_API_KEY") is None:
        dotenv.load_dotenv()

    assert (
        os.getenv("OPENAI_API_KEY") is not None
    ), "OpenAI API key must be set in `.env`."

    if (
        not cache_non_deterministic_ok
        and "temperature" in kwargs
        and kwargs["temperature"] > 0
    ):
        raise ValueError("Responses for temperature > 0 shouldn't be cached.")

    return openai.ChatCompletion.create(**kwargs)


@typechecked
def get_file_id(fname: str | Path) -> str:
    """Get the file ID from OpenAI API."""

    if isinstance(fname, str):
        fname = Path(fname)

    if not fname.is_file():
        raise FileNotFoundError(f"{fname} doesn't exist.")

    response = openai.File.list()

    for file in response.data:
        if file.filename == fname:
            return file.id

    response = openai.File.create(
        file=open(fname, "rb"),
        purpose="fine-tune",
    )

    return response.id


def get_fine_tuned_model(job_id: str) -> str:
    """Get the model ID from OpenAI API."""

    if not isinstance(job_id, str):
        raise TypeError("Expected job_id to be a string.")

    response = openai.FineTune.list()

    for model in response.data:
        if model.id != job_id:
            continue

        if model.fine_tuned_model is None:
            raise ValueError(f"Job ID {job_id} doesn't have a fine-tuned model.")

        return model.fine_tuned_model

    raise ValueError(f"Job ID {job_id} not found.")


def wait_for_fine_tune_completion(fine_tune_id: str) -> None:
    """Wait for a fine-tuning job to complete."""

    while True:
        response = openai.FineTune.retrieve(fine_tune_id)
        status = response.status

        if status in ["succeeded", "failed", "cancelled"]:
            return print(f"Fine-tuning job {fine_tune_id} {status}.")

        print(f"Fine-tuning job {fine_tune_id} is {status}. Waiting...")
        time.sleep(60)


@functools.lru_cache(maxsize=1)
def get_homophone_lookup_table(fname: str = "datasets/homophones.txt") -> Dict:
    """Gets a list of homophone pairs from a file."""
    with open(fname, "r") as f:
        homophone_list = f.read().splitlines()

    homophone_list = [homophone.split(" ") for homophone in homophone_list]
    homophone_list = [
        [w for w in homophone if w.isalpha()] for homophone in homophone_list
    ]
    homophone_list = [homophone for homophone in homophone_list if len(homophone) > 1]

    lookup_table = {}

    for homophones in homophone_list:
        for word in homophones:
            for other_word in homophones:
                if word == other_word:
                    continue

                if word not in lookup_table:
                    lookup_table[word] = ()

                lookup_table[word] += (other_word,)

    return lookup_table


assert get_homophone_lookup_table().get("bird") == ("burred",)
assert get_homophone_lookup_table().get("soot") == ("suit",)
assert get_homophone_lookup_table().get("rot") == ("wrought",)
assert get_homophone_lookup_table().get("tai") == ("taille", "tie", "tye")


@functools.lru_cache(maxsize=1)
def get_synonyms(fname: str = "datasets/thesaurus.jsonl") -> Dict:
    """Loads synonyms from thesaurus at `fname`. Note: this is now depreciated. Use `get_synonym` instead."""
    # pre-conditions
    assert os.path.exists(fname), f"File {fname} does not exist"

    # load synonyms
    jsonls = open(fname, "r").read().splitlines()
    jsonls = [json.loads(s) for s in jsonls]
    jsonls = filter(lambda jl: " " not in jl["word"], jsonls)
    jsonls = map(lambda jl: {"word": jl["word"], "synonyms": jl["synonyms"]}, jsonls)
    jsonls = map(lambda jl: {**jl, "word": jl["word"].lower()}, jsonls)
    jsonls = map(
        lambda jl: {
            **jl,
            "word": filter(lambda s: all(c.isalpha() for c in s), jl["word"]),
        },
        jsonls,
    )
    jsonls = map(lambda jl: {**jl, "word": "".join(list(jl["word"]))}, jsonls)
    jsonls = map(
        lambda jl: {**jl, "synonyms": map(lambda s: s.lower(), jl["synonyms"])}, jsonls
    )
    jsonls = map(
        lambda jl: {
            **jl,
            "synonyms": filter(lambda s: all(c.isalpha() for c in s), jl["synonyms"]),
        },
        jsonls,
    )
    jsonls = map(
        lambda jl: {
            **jl,
            "synonyms": filter(lambda s: s != jl["word"], jl["synonyms"]),
        },
        jsonls,
    )
    jsonls = map(
        lambda jl: {**jl, "synonyms": filter(lambda s: " " not in s, jl["synonyms"])},
        jsonls,
    )
    jsonls = map(lambda jl: {**jl, "synonyms": tuple(jl["synonyms"])}, jsonls)
    jsonls = functools.reduce(
        lambda acc, jl: {**acc, jl["word"]: jl["synonyms"]}, jsonls, {}
    )
    synonyms = jsonls

    # post-conditions
    assert isinstance(synonyms, dict), "Synonyms must be a dictionary"
    assert len(synonyms) != 0, "Synonyms must be non-empty"

    return synonyms


def get_synonym(word: str) -> str:
    """Gets a synonym of the word."""
    # pre-conditions
    assert isinstance(word, str), f"Word must be a string, but got {word}."
    assert word != "", "Word can't be the empty string."

    # body
    prompt = (
        "A synonym of 'angry' is 'irritated'.\n"
        "A synonym of 'grass' is 'turf'.\n"
        "A synonym of 'car' is 'automobile'.\n"
        "A synonym of 'exam' is 'test'.\n"
        f"A synonym of '{word}' is '"
    )

    completions = openai_completion_create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
    )

    assert "choices" in completions, "No choices in completion"
    assert len(completions["choices"]) >= 1, "Need at least one choice in completion"
    assert "text" in completions["choices"][0], "No text in completion choice"

    completion = completions["choices"][0]["text"]

    assert completion != "", "Completion must not be an empty string."
    assert completion.endswith("'."), f"Expected completion to end with `'.`."

    synonym = completion[:-2]

    # post-conditions
    assert isinstance(synonym, str), f"Synonym must be a string, but got {synonym}."
    assert synonym != word, "Synonym must be different from the word."
    assert synonym != "", "Synonym can't be the empty string."

    return synonym


assert get_synonym("that") == "those"
assert get_synonym("count") == "calculate"
assert get_synonym("record") == "document"


def paraphrase(
    sentence: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
) -> str:
    """Paraphrases the sentence. Only supports `gpt-3.5-turbo` and `gpt-4`."""
    # pre-conditions
    assert isinstance(sentence, str), f"Word must be a string, but got {sentence}."
    assert sentence != "", "Word can't be the empty string."
    assert model in ["gpt-3.5-turbo", "gpt-4"], "Model not supported."

    # body
    stop_sequence = "</rewording>"
    preamble = (
        "Your task is to paraphrase given pieces of text. Please end your responses"
        f' with "{stop_sequence}" to clearly mark the end of the response.'
    )
    prompt = (
        "<original>The boy discovered the lost treasure after many years of"
        " searching</original><rewording>After years of exploration, the young man"
        " finally found the long-lost treasure</rewording>\n<original>Exercising"
        " regularly is a great way to stay fit and"
        " healthy</original><rewording>Maintaining a regular workout regimen is an"
        " effective strategy for preserving health and"
        " fitness</rewording>\n<original>Climate change is a serious issue that needs"
        " immediate attention</original><rewording>The matter of climate change is"
        " severe and calls for prompt focus and action</rewording>\n<original>He quit"
        " his job because he didnt like his boss</original><rewording>He resigned from"
        " his position as he disliked his"
        f" superior</rewording>\n<original>{sentence}</original><rewording>"
    )

    response = openai_chatcompletion_create(
        model=model,
        messages=[
            {"role": "system", "content": preamble},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=32,
        cache_non_deterministic_ok=True,
    )

    completion: str = response["choices"][0]["message"]["content"]

    assert completion != "", "Completion must not be an empty string."
    assert completion.endswith(
        stop_sequence
    ), f"Expected completion to end with `{stop_sequence}`. Got: '{completion}'"

    synonym = completion[: -len(stop_sequence)]

    # post-conditions
    assert isinstance(synonym, str), f"Synonym must be a string, but got {synonym}."
    assert synonym != sentence, "Synonym must be different from the word."
    assert synonym != "", "Synonym can't be the empty string."

    return synonym


assert (
    paraphrase("She took her umbrella because the forecast predicted rain")
    == "She brought her umbrella due to the forecast indicating precipitation"
), f"Got: {paraphrase('She took her umbrella because the forecast predicted rain')}"


def translate(word: str, language: str) -> str:
    """Translates `word` to `language`. Automatically detects the language of `word`."""

    # pre-conditions
    assert isinstance(word, str), f"Word must be a string, but got `{word}`."
    assert isinstance(
        language, str
    ), f"Language must be a string, but got `{language}`."
    assert word != "", f"Word must not be an empty string."
    assert word.isalpha(), f"Word must be alphabetic to be translated."
    assert language != "", f"Language must not be an empty string."

    # body
    prompt = (
        "The Japanese translation of 'sofa' is 'ソファー'.\n"
        "The Portuguese translation of 'sky' is 'céu'.\n"
        "The Russian translation of 'wall' is 'стена'.\n"
        f"The {language} translation of '{word}' is '"
    )
    completions = openai_completion_create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
    )

    assert "choices" in completions, "No choices in completion"
    assert len(completions["choices"]) >= 1, "Need at least one choice in completion"
    assert "text" in completions["choices"][0], "No text in completion choice"

    completion = completions["choices"][0]["text"]

    assert completion != "", "Completion must not be an empty string."
    assert completion.endswith("'."), f"Expected completion to end with `'.`."

    translation = completion[:-2]

    # post-conditions
    assert isinstance(
        translation, str
    ), f"Translated word must be a string, but got `{translation}`."
    assert translation != "", f"Translated word must not be an empty string."

    return translation


assert translate("dog", "Vietnamese") == "chó", f"Got {translate('dog', 'Vietnamese')}"
assert translate("dog", "Turkish") == "köpek", f"Got {translate('dog', 'Turkish')}"
assert translate("dog", "Japanese") == "犬", f"Got {translate('dog', 'Japanese')}"


def misspell(word: str) -> str:
    """Returns a common misspelling of `word`."""

    # pre-conditions
    assert isinstance(word, str), f"Word must be a string, but got `{word}`."
    assert word != "", f"Word must not be an empty string."
    assert word.isalpha(), f"Word must be alphabetic to be misspelled."

    # body
    prompt = (
        "A common misspelling of 'acquire' is 'aquire'.\n"
        "A common misspelling of 'consensus' is 'concensus'.\n"
        "A common misspelling of 'entrepreneur' is 'entrepeneur'.\n"
        "A common misspelling of 'license' is 'licence'.\n"
        f"A common misspelling of '{word}' is '"
    )
    completions = openai_completion_create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
    )

    assert "choices" in completions, "No choices in completion"
    assert len(completions["choices"]) >= 1, "Need at least one choice in completion"
    assert "text" in completions["choices"][0], "No text in completion choice"

    completion = completions["choices"][0]["text"]

    assert completion != "", "Completion must not be an empty string."
    assert completion.endswith("'."), f"Expected completion to end with `'.`."

    misspelling = completion[:-2]

    # post-conditions
    assert isinstance(
        misspelling, str
    ), f"Misspelling must be a string, but got `{misspelling}`."
    assert misspelling != "", f"Misspelling must not be an empty string."

    return misspelling


assert misspell("loud") == "loude"
assert misspell("liaison") == "liason"
assert misspell("privilege") == "priviledge"


def different_part_of_speech(word: str) -> str:
    """Returns a word that has the same meaning as `word` but is a different part of speech."""
    # pre-conditions
    assert isinstance(word, str), f"Word must be a string, but got {word}."
    assert word != "", "Word can't be the empty string."

    # body
    prompt = (
        "Given a word, generate a word with a different part of speech.\n\n"
        "A different part of speech for the word 'run' is 'running'.\n"
        "A different part of speech for the word 'Paris' is 'Parisian'.\n"
        "A different part of speech for the word 'love' is 'loving'.\n"
        "A different part of speech for the word 'happy' is 'happiness'.\n"
        f"A different part of speech for the word '{word}' is '"
    )

    completions = openai_completion_create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=16,
        temperature=0.0,
    )

    assert "choices" in completions, "No choices in completion"
    assert len(completions["choices"]) >= 1, "Need at least one choice in completion"
    assert "text" in completions["choices"][0], "No text in completion choice"

    completion = completions["choices"][0]["text"]

    assert completion != "", "Completion must not be an empty string."
    assert completion.endswith("'."), f"Expected completion to end with `'.`."

    new_word = completion[:-2]

    # post-conditions
    assert isinstance(
        new_word, str
    ), f"New word must be a string, but got `{new_word}`."
    assert new_word != word, f"New word `{new_word}` must be different from `{word}`."
    assert new_word != "", "New word must not be an empty string."

    return new_word


assert different_part_of_speech("care") == "caring"
assert different_part_of_speech("hike") == "hiker"
assert different_part_of_speech("astound") == "astounding"


def is_equivalent(
    articulation: str, other_articulation: str, seed: int, model: str = "gpt-4"
) -> bool:
    df = DF_LABELED_ARTICULATIONS.sample(n=8, replace=False, random_state=seed)
    preamble = (
        "You're a knowledgeable and thorough data labeller who's been tasked with"
        " creating a labelled dataset. The dataset consists of labelled input pairs,"
        " each of which articulates a binary text classification rule. Your job is to"
        " generate the label these input pairs 'True' if they articulate equivalent"
        " rules and 'False' otherwise."
    )
    prompt = ""

    for index, row in df.iterrows():
        expected_articulation = row["expected_articulation"]
        actual_articulation = row["actual_articulation"]
        is_equivalent = row["is_equivalent"]

        prompt += f"Input 1: {expected_articulation}\n"
        prompt += f"Input 2: {actual_articulation}\n"
        prompt += f"Label:  {is_equivalent}"

    prompt += f"Input 1: {articulation}\n"
    prompt += f"Input 2: {other_articulation}\n"
    prompt += f"Label:"

    response = openai_chatcompletion_create(
        model=model,
        messages=[
            {"role": "system", "content": preamble},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1,
    )

    completion: str = response["choices"][0]["message"]["content"]

    return completion.strip() == "True"


assert is_equivalent(
    "The input does not contain the word `slow`",
    "The pattern is that the inputs labelled 'True' do not contain the word 'slow',"
    " while the inputs labelled 'False' contain the word 'slow'.",
    0,
)

assert not is_equivalent(
    "The input repeats the word `collect`",
    "The pattern is that the input is labelled 'True' if it contains the word 'collect'"
    " at least three times, and 'False' otherwise.",
    0,
)


def jumble_case(word):
    """Randomly changes the case of characters in a word."""
    # pre-conditions
    if not isinstance(word, str):
        raise TypeError("Input word must be a string.")

    if len(word) == 0:
        raise ValueError("Input word cannot be an empty string.")

    # body
    flip = lambda c: c.upper() if c.islower() else c.lower()
    n_change = random.randint(1, len(word))
    idxs = random.sample(range(len(word)), k=n_change)
    new_chars = [flip(word[i]) if i in idxs else word[i] for i in range(len(word))]
    new_word = "".join(new_chars)

    # post-conditions
    assert isinstance(new_word, str)
    assert len(new_word) == len(word)

    return new_word


def contains_subset(collection: Collection[Collection]) -> bool:
    """Checks if any element is a subset of another element."""
    # pre-conditions
    assert isinstance(collection, Collection), "Expected input to be a collection."
    assert all(
        isinstance(element, Collection) for element in collection
    ), "Expected all elements to be collections."

    # body
    for x in collection:
        for y in collection:
            if x == y:
                continue

            if x in y:
                return True

    return False


def filter_subsets(collection: Collection[Collection[T]]) -> Tuple[Collection[T]]:
    """Filters out any elements that are subsets of other elements."""
    # pre-conditions
    assert isinstance(collection, Collection), "Expected input to be a collection."
    assert all(
        isinstance(element, Collection) for element in collection
    ), "Expected all elements to be collections."

    # body
    new_collection = set()

    for x in collection:
        for y in collection:
            if x == y:
                continue

            if x in y:
                break
        else:
            new_collection.add(x)

    new_collection = tuple(new_collection)

    # post-conditions
    assert isinstance(new_collection, tuple), "Expected output to be a set."
    assert len(new_collection) <= len(
        collection
    ), "Expected new collection to be no bigger than the original."

    return new_collection


def get_rule_features(rule) -> tuple[str]:
    """Gets features used in rule."""

    fields = rule.generation_behaviour.__dict__
    features = tuple()

    for field in fields:
        if field == "features":
            features += fields[field]

        if field == "feature":
            features += (fields[field],)

        if field == "first_feature":
            features += (fields[field],)

        if field == "second_feature":
            features += (fields[field],)

        if field == "w":
            features += (fields[field],)

        if field == "w_prime":
            features += (fields[field],)

        if field == "rule":
            features += get_rule_features(fields[field])

        if field == "rules":
            for rule in fields[field]:
                features += get_rule_features(rule)

    return features


def tiktoken_encode(self, *args, **kwargs):
    """Custom call method for tiktoken Encoding class. Makes it compatible with transformers API."""
    assert len(args) >= 1, "Expected at least one argument to tokenizer call."

    text = args[0]

    if isinstance(text, list):
        return self.encode_batch(*args, **kwargs)

    return self.encode(*args, **kwargs)


def xor(array: Tuple[bool, ...]) -> bool:
    """Returns the exclusive or of the elements in `array`."""
    assert len(array) >= 1, "Expected at least one argument"
    assert all(isinstance(x, bool) for x in array), "Expected all elements to be bools"

    out = sum(array) == 1

    assert isinstance(out, bool), "Expected output to be bool"

    return out


def get_actual_and_expected_labels(
    completion: openai.Completion,
) -> Tuple[Tuple[str], Tuple[str]]:
    """Returns a tuple of (actual, expected) labels in the prompt. Assumes completion is using the default prompt format."""

    assert "choices" in completion, "No choices in completion"
    assert len(completion["choices"]) == 1, "Expected exactly one choice in completion"
    assert "logprobs" in completion["choices"][0], "No `logprobs` in completion"
    assert "tokens" in completion["choices"][0]["logprobs"], "No `tokens` in completion"
    assert (
        "top_logprobs" in completion["choices"][0]["logprobs"]
    ), "No `top_logprobs` in completion"

    tokens = completion["choices"][0]["logprobs"]["tokens"]
    top_logprobs = completion["choices"][0]["logprobs"]["top_logprobs"]

    assert all(
        len(logprobs) == 1 for logprobs in top_logprobs[1:]
    ), "This function assumes each token in the completion has exactly one logprob."

    assert "usage" in completion, "No `usage` in completion"
    assert "prompt_tokens" in completion["usage"], "No `prompt_tokens` in completion"

    prompt_tokens = completion["usage"]["prompt_tokens"]

    actual = []
    expected = []

    for i in range(2, len(tokens)):
        if i >= prompt_tokens:
            break  # end of the prompt; actual tokens are now generated and not ground truth

        if tokens[i - 2] != "Label" or tokens[i - 1] != ":":
            continue

        expected_token = tokens[i]
        [actual_token] = top_logprobs[i].keys()

        actual.append(actual_token)
        expected.append(expected_token)

    return tuple(actual), tuple(expected)


def get_actual_and_expected_multiple_choice_answer(
    completion: openai.Completion,
) -> Tuple[str, str]:
    """Returns a tuple of (actual, expected) mulitple choice answer. Assumes completion is using the default prompt format."""

    assert "choices" in completion, "No choices in completion"
    assert len(completion["choices"]) == 1, "Expected exactly one choice in completion"
    assert "logprobs" in completion["choices"][0], "No `logprobs` in completion"
    assert "tokens" in completion["choices"][0]["logprobs"], "No `tokens` in completion"
    assert (
        "top_logprobs" in completion["choices"][0]["logprobs"]
    ), "No `top_logprobs` in completion"

    tokens = completion["choices"][0]["logprobs"]["tokens"]
    top_logprobs = completion["choices"][0]["logprobs"]["top_logprobs"]

    assert all(
        len(logprobs) == 1 for logprobs in top_logprobs[1:]
    ), "This function assumes each token in the completion has exactly one logprob."

    assert "usage" in completion, "No `usage` in completion"
    assert "prompt_tokens" in completion["usage"], "No `prompt_tokens` in completion"

    prompt_tokens = completion["usage"]["prompt_tokens"]

    for i in range(2, len(tokens)):
        if i >= prompt_tokens:
            break  # end of the prompt; actual tokens are now generated and not ground truth

        if tokens[i - 3] != "Answer" or tokens[i - 2] != ":":
            continue

        [actual] = top_logprobs[i].keys()
        expected = tokens[i]

        return actual, expected

    assert False, "No answer found in completion"


def latexify(s: str, words: str | Collection[str]) -> str:
    """
    Replaces every occurance of `word` in `s` with `$word$`. Optionally pass in a single word.
    """
    if isinstance(words, str):
        words = [words]

    for w in words:
        s = s.replace(f" {w} ", f" ${w}$ ")

        if s.endswith(f" {w}"):
            s = s[: -len(w)] + f" ${w}$"

    return s


def make_readable(var_name: str) -> str:
    """
    Makes a variable name readable (e.g. "contains_the_word_w_prime" -> "Contains the word W'").
    """
    split = var_name.split("_")
    new_split = []

    i = 0

    while i < len(split):
        if i == 0:
            word = split[i].title()
        elif split[i] == "french":
            word = split[i].title()
        elif i + 1 < len(split) and split[i] == "w" and split[i + 1] == "prime":
            word = "W'"
            i += 1
        elif i + 1 < len(split) and split[i] == "k" and split[i + 1] == "prime":
            word = "k'"
            i += 1
        elif split[i] == "w":
            word = "W"
        else:
            word = split[i]

        new_split.append(word)
        i += 1

    readable = " ".join(new_split)

    return latexify(readable, ["W", "W'", "k", "k'", "i", "x"])


def to_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a dataframe with at least columns [rule, is_correct, model_name] into a DataFrame with columns Rule and Model Accuracy for each model in the dataframe. Include the mean and standard error of the mean for each rule. Convert to percentages. Stitch standard error together with mean with a plusminus sign. Round to 1 decimal place.
    """
    assert "rule" in df.columns, "No rule column"
    assert "is_correct" in df.columns, "No is_correct column"
    assert "model_name" in df.columns, "No model_name column"

    model_names = df["model_name"].unique()
    model_dfs = [df[df["model_name"] == model_name] for model_name in model_names]
    new_dfs = []

    for model_name, model_df in zip(model_names, model_dfs):
        assert (
            model_df["model_name"].nunique() == 1
        ), "Multiple model names in dataframe"

        model_df = model_df.drop(columns=["model_name"])
        model_df = (
            model_df[["rule", "is_correct"]]
            .groupby(["rule"])
            .agg(["mean", "sem"])
            .reset_index()
        )
        model_df.columns = model_df.columns.map("".join)
        model_df["mean"] = model_df["is_correctmean"] * 100
        model_df["sem"] = model_df["is_correctsem"] * 100
        model_df = model_df[["rule", "mean", "sem"]]
        model_df["mean"] = model_df["mean"].round(1)
        model_df["sem"] = model_df["sem"].round(1)
        # model_df = model_df.sort_values(by="mean", ascending=False)
        model_df["mean"] = model_df["mean"].astype(str)
        model_df["sem"] = model_df["sem"].astype(str)
        model_df["mean_and_sem"] = model_df["mean"] + "\pm" + model_df["sem"]
        model_df = model_df[["rule", "mean_and_sem"]]
        model_df.columns = ["Rule", model_name]
        model_df[model_name] = "$" + model_df[model_name] + "$"
        model_df["Rule"] = model_df["Rule"].apply(make_readable)
        model_df = model_df.set_index("Rule")
        new_dfs.append(model_df)

    return pd.concat(new_dfs, axis=1).reset_index(drop=False)


def to_latex(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame into a latex table.
    """
    return df.style.hide(axis="index").to_latex(hrules=True)


def to_accuracy_pm_error(
    df: pd.DataFrame, acc_col: str, err_col: str, scale_by: float = 100.0
) -> pd.Series:
    """
    Creates a new column with the accuracy and error in the format `$acc \pm err$`.
    """
    assert acc_col in df.columns, f"Column `{acc_col}` not in dataframe!"
    assert err_col in df.columns, f"Column `{err_col}` not in dataframe!"

    df = df[[acc_col, err_col]].copy()

    df[acc_col] = df[acc_col] * scale_by
    df[err_col] = df[err_col] * scale_by

    df[acc_col] = df[acc_col].round(1)
    df[err_col] = df[err_col].round(1)

    df[acc_col] = df[acc_col].astype(str)
    df[err_col] = df[err_col].astype(str)

    series = "$" + df[acc_col] + "\pm" + df[err_col] + "$"

    return series


def create_hash(s: str) -> str:
    """Generates a SHA256 hash from a given input string."""
    enc = s.encode("utf-8")
    obj = hashlib.sha256()
    obj.update(enc)
    return obj.hexdigest()
