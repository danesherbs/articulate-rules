import utils
import datasets

from typing import Tuple


##################
# CRISP FEATURES #
##################


# Contains the word W
def contains_the_word_w(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = datasets.ContainsFeatureGenerationBehaviour(
        feature=w,
        words=features,
        n_words_per_example=n_words_per_example,
    )

    discrimination_behaviour = (
        datasets.ContainsSpaceSeparatedFeatureDiscriminationBehaviour(
            word=w,
            min_count=1,
            max_count=n_words_per_example,
        )
    )

    return datasets.Rule(
        explanation=f"The input contains the word `{w}`",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Does not contain the word W
def does_not_contain_the_word_w(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = datasets.DoesNotContainFeatureGenerationBehaviour(
        feature=w,
        words=features,
        n_words_per_example=n_words_per_example,
    )

    discrimination_behaviour = (
        datasets.DoesNotContainSpaceSeparatedFeatureDiscriminationBehaviour(word=w)
    )

    return datasets.Rule(
        explanation=f"The input does not contain the word `{w}`",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Contains the words W and W’
def contains_the_words_w_and_w_prime(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.ConjunctionRule(
        explanation=f"The input contains the word `{w}` and `{w_prime}`",
        rules=[
            contains_the_word_w(w, features, n_words_per_example),
            contains_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Contains the words W or W’
def contains_the_words_w_or_w_prime(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.DisjunctionRule(
        explanation=f"The input contains the word `{w}` or `{w_prime}`",
        rules=[
            contains_the_word_w(w, features, n_words_per_example),
            contains_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Contains W or W’ but not both
def contains_the_words_w_or_w_prime_but_not_both(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.XorRule(
        explanation=f"The input contains the word `{w}` or `{w_prime}` but not both",
        rules=[
            contains_the_word_w(w, features, n_words_per_example),
            contains_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Contains W and a digit
def contains_the_word_w_and_a_digit(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.ConjunctionRule(
        explanation=f"The input contains the word `{w}` and a digit",
        rules=[
            contains_the_word_w(w, features, n_words_per_example),
            contains_a_digit(features, n_words_per_example),
        ],
    )


# Contains W and a French word
def contains_the_word_w_and_a_french_word(
    w: str,
    features: Tuple[str],
    french_words: Tuple[str],
    n_words_per_example: int,
    **kwargs,
):
    return datasets.ConjunctionRule(
        explanation=f"The input contains the word `{w}` and a French word",
        rules=[
            contains_the_word_w(w, features, n_words_per_example),
            contains_a_french_word(features, french_words, n_words_per_example),
        ],
    )


# Contains the word W and words are in alphabetical order
def contains_the_word_w_and_is_in_alphabetical_order(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.ConjunctionRule(
        explanation=f"The input contains the word `{w}` and is in alphabetical order",
        rules=[
            contains_the_word_w(w, features, n_words_per_example),
            words_are_in_alphabetical_order(features, n_words_per_example),
        ],
    )


# Contains a digit
def contains_a_digit(features: Tuple[str], n_words_per_example: int, **kwargs):
    new_features = features + tuple(str(n) for n in range(10))

    generation_behaviour = (
        datasets.SampledWordsContainAnySpaceSeparatedFeaturesGenerationBehaviour(
            features=tuple(str(n) for n in range(10)),
            words=new_features,
            n_words_per_example=n_words_per_example,
        )
    )

    discrimination_behaviour = (
        datasets.ContainsAnySpaceSeparatedFeatureDiscriminationBehaviour(
            words=tuple(str(n) for n in range(10)),
        )
    )

    return datasets.Rule(
        explanation=f"The input contains a digit",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Contains a French word
def contains_a_french_word(
    features: Tuple[str], french_words: Tuple[str], n_words_per_example: int, **kwargs
):
    new_features = features + french_words

    generation_behaviour = (
        datasets.SampledWordsContainAnySpaceSeparatedFeaturesGenerationBehaviour(
            features=french_words,
            words=new_features,
            n_words_per_example=n_words_per_example,
        )
    )

    discrimination_behaviour = (
        datasets.ContainsAnySpaceSeparatedFeatureDiscriminationBehaviour(
            words=french_words,
        )
    )

    return datasets.Rule(
        explanation=f"The input contains a French word",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Contains W between k and k’ times
def contains_the_word_w_between_k_and_k_prime_times(
    w: str,
    k: int,
    k_prime: int,
    features: Tuple[str],
    n_words_per_example: int,
    **kwargs,
):
    generation_behaviour = (
        datasets.SampledWordsAppearCertainNumberOfTimesGenerationBehaviour(
            feature=w,
            words=features,
            min_feature_count=k,
            max_feature_count=k_prime,
            n_words_per_example=n_words_per_example,
        )
    )

    discrimination_behaviour = (
        datasets.ContainsSpaceSeparatedFeatureDiscriminationBehaviour(
            word=w,
            min_count=k,
            max_count=k_prime,
        )
    )

    return datasets.Rule(
        explanation=(
            f"The input contains the word `{w}` between {k} and {k_prime} times"
        ),
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Contains W at least K times
def contains_the_word_w_at_least_k_times(
    w: str, k: int, features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = (
        datasets.SampledWordsAppearCertainNumberOfTimesGenerationBehaviour(
            feature=w,
            words=features,
            min_feature_count=k,
            max_feature_count=n_words_per_example,
            n_words_per_example=n_words_per_example,
        )
    )

    discrimination_behaviour = (
        datasets.ContainsSpaceSeparatedFeatureDiscriminationBehaviour(
            word=w,
            min_count=k,
            max_count=n_words_per_example,
        )
    )

    return datasets.Rule(
        explanation=f"The input contains the word `{w}` at least {k} times",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Contains the word W within X positions of the word W'
def contains_the_word_w_within_x_positions_of_the_word_w_prime(
    w: str,
    w_prime: str,
    features: Tuple[str],
    x: int,
    n_words_per_example: int,
    **kwargs,
):
    generation_behaviour = (
        datasets.SampledWordsAreWithinXPositionsOfEachOtherGenerationBehaviour(
            w=w,
            w_prime=w_prime,
            words=features,
            x=x,
            n_words_per_example=n_words_per_example,
        )
    )

    discrimination_behaviour = (
        datasets.SampledWordsAreWithinXPositionsOfEachOtherDiscriminationBehaviour(
            x=x,
            w=w,
            w_prime=w_prime,
        )
    )

    return datasets.Rule(
        explanation=(
            f"The input contains the word `{w}` within {x} positions of the word"
            f" `{w_prime}`"
        ),
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Repeats the word W
def repeats_the_word_w(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = (
        datasets.SampledWordsAppearCertainNumberOfTimesGenerationBehaviour(
            feature=w,
            words=features,
            min_feature_count=2,
            max_feature_count=n_words_per_example,
            n_words_per_example=n_words_per_example,
        )
    )

    discrimination_behaviour = (
        datasets.ContainsSpaceSeparatedFeatureDiscriminationBehaviour(
            word=w,
            min_count=2,
            max_count=n_words_per_example,
        )
    )

    return datasets.Rule(
        explanation=f"The input repeats the word `{w}`",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Repeats the word W and contains a French word
def repeats_the_word_w_and_contains_a_french_word(
    w: str,
    features: Tuple[str],
    french_words: Tuple[str],
    n_words_per_example: int,
    **kwargs,
):
    return datasets.ConjunctionRule(
        explanation=f"Repeats the word `{w}` and a French word",
        rules=[
            repeats_the_word_w(w, features, n_words_per_example),
            contains_a_french_word(features, french_words, n_words_per_example),
        ],
    )


# Repeats the word W and words are in alphabetical order
def repeats_the_word_w_and_is_in_alphabetical_order(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.ConjunctionRule(
        explanation=f"Repeats the word `{w}` and is in alphabetical order",
        rules=[
            repeats_the_word_w(w, features, n_words_per_example),
            words_are_in_alphabetical_order(features, n_words_per_example),
        ],
    )


# Repeats the word W and contains a digit
def repeats_the_word_w_and_contains_a_digit(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.ConjunctionRule(
        explanation=f"Repeats the word `{w}` and contains a digit",
        rules=[
            repeats_the_word_w(w, features, n_words_per_example),
            contains_a_digit(features, n_words_per_example),
        ],
    )


# Repeats the word W and starts with the word W
def repeats_the_word_w_and_starts_with_w(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.ConjunctionRule(
        explanation=f"Repeats the word `{w}` and starts with the word `{w}`",
        rules=[
            starts_with_the_word_w(w, features, n_words_per_example),
            repeats_the_word_w(w, features, n_words_per_example),
        ],
    )


# Repeats the word W and contains the word W'
def repeats_the_word_w_and_contains_the_word_w_prime(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.ConjunctionRule(
        explanation=f"Repeats the word `{w}` and contains the word `{w_prime}`",
        rules=[
            repeats_the_word_w(w, features, n_words_per_example),
            contains_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Words are in alphabetical order
def words_are_in_alphabetical_order(
    features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = datasets.SampledWordsAreSortedGenerationBehaviour(
        words=features,
        n_words_per_example=n_words_per_example,
        is_descending=False,
    )

    discrimination_behaviour = (
        datasets.WordsAreSortedAlphabeticallyDiscriminationBehaviour(
            is_descending=False,
        )
    )

    return datasets.Rule(
        explanation="The words in the input are in alphabetical order",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Words are in reverse alphabetical order
def words_are_in_reverse_alphabetical_order(
    features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = datasets.SampledWordsAreSortedGenerationBehaviour(
        words=features,
        n_words_per_example=n_words_per_example,
        is_descending=True,
    )

    discrimination_behaviour = (
        datasets.WordsAreSortedAlphabeticallyDiscriminationBehaviour(
            is_descending=True,
        )
    )

    return datasets.Rule(
        explanation="The words in the input are in reverse alphabetical order",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# W appears in position i
def word_w_appears_in_position_i(
    w: str, i: int, features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = datasets.ContainsFeatureGenerationBehaviour(
        feature=w,
        words=features,
        n_words_per_example=n_words_per_example,
        allowed_feature_idxs=(i,),
    )

    discrimination_behaviour = (
        datasets.ContainsSpaceSeparatedFeatureDiscriminationBehaviour(
            word=w,
            min_count=1,
            max_count=n_words_per_example,
            feature_must_appear_idxs=(i,),
        )
    )

    return datasets.Rule(
        explanation=f"The word `{w}` is in position {i}",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Starts with W
def starts_with_the_word_w(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return word_w_appears_in_position_i(w, 0, features, n_words_per_example)


# Does not start with W
def does_not_start_with_the_word_w(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.NegationRule(
        explanation=f"The input does not start with the word `{w}`",
        rule=starts_with_the_word_w(w, features, n_words_per_example),
    )


# Ends with W
def ends_with_the_word_w(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return word_w_appears_in_position_i(
        w, n_words_per_example - 1, features, n_words_per_example
    )


# Starts with W or contains W’
def starts_with_the_word_w_or_contains_the_word_w_prime(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.DisjunctionRule(
        explanation=(
            f"The input starts with the word `{w}` or contains the word `{w_prime}`"
        ),
        rules=[
            starts_with_the_word_w(w, features, n_words_per_example),
            contains_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Starts with W or repeats W’
def starts_with_the_word_w_or_repeats_the_word_w_prime(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.DisjunctionRule(
        explanation=(
            f"The input starts with the word `{w}` or repeats the word `{w_prime}`"
        ),
        rules=[
            starts_with_the_word_w(w, features, n_words_per_example),
            repeats_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Starts with W or contains W’, but not both
def starts_with_the_word_w_or_contains_the_word_w_prime_but_not_both(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.XorRule(
        explanation=(
            f"The input starts with the word `{w}` or contains the word `{w_prime}`,"
            " but not both"
        ),
        rules=[
            starts_with_the_word_w(w, features, n_words_per_example),
            contains_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Starts with W or repeats W’, but not both
def starts_with_the_word_w_or_repeats_the_word_w_prime_but_not_both(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.XorRule(
        explanation=(
            f"The input starts with the word `{w}` or repeats the word `{w_prime}`, but"
            " not both"
        ),
        rules=[
            starts_with_the_word_w(w, features, n_words_per_example),
            repeats_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Starts or ends with W but not both
def starts_or_ends_with_the_word_w_but_not_both(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.XorRule(
        explanation=f"The input starts or ends with the word `{w}`, but not both",
        rules=[
            starts_with_the_word_w(w, features, n_words_per_example),
            word_w_appears_in_position_i(
                w, n_words_per_example - 1, features, n_words_per_example
            ),
        ],
    )


# Ends with W or contains W’
def ends_with_the_word_w_or_contains_the_word_w_prime(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    return datasets.DisjunctionRule(
        explanation=(
            f"The input ends with the word `{w}` or contains the word `{w_prime}`"
        ),
        rules=[
            ends_with_the_word_w(w, features, n_words_per_example),
            contains_the_word_w(w_prime, features, n_words_per_example),
        ],
    )


# Both W and W' appear and appear in sorted order
def contains_the_word_w_and_the_word_w_prime_in_sorted_order(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = datasets.WordsAreOrderedGenerationBehaviour(
        features=(w, w_prime),
        words=features,
        n_words_per_example=n_words_per_example,
        is_descending=False,
    )

    discrimination_behaviour = datasets.WordsAreOrderedDiscriminationBehaviour(
        features=(w, w_prime),
        is_descending=False,
    )

    return datasets.Rule(
        explanation=(
            f"The input contains the words `{w}` and `{w_prime}` in alphabetical order"
        ),
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


# Contains W followed by W’
def contains_the_word_w_followed_by_w_prime(
    w: str, w_prime: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    generation_behaviour = (
        datasets.ContainsWordFollowedByAnotherWordGenerationBehaviour(
            first_feature=w,
            second_feature=w_prime,
            words=features,
            n_words_per_example=n_words_per_example,
        )
    )

    discrimination_behaviour = (
        datasets.ContainsWordFollowedByAnotherWordDiscriminationBehaviour(
            first_feature=w,
            second_feature=w_prime,
        )
    )

    return datasets.Rule(
        explanation=f"The input contains the word `{w}` before the word `{w_prime}`",
        generation_behaviour=generation_behaviour,
        discrimination_behaviour=discrimination_behaviour,
    )


#################################
# ADVERSARIAL ATTACK HYPOTHESIS #
#################################


def contains_the_word_w_or_a_synonym(
    w: str, features: Tuple[str], n_words_per_example: int, **kwargs
):
    synonym = utils.get_synonym(w)
    features_with_synonym = features + (synonym,)

    assert synonym.isalpha(), f"Synonym should only be letters. Got `{synonym}`."

    if synonym == w:
        return contains_the_word_w(w, features, n_words_per_example)

    return datasets.DisjunctionRule(
        explanation=f"The input contains the word `{w}` or a synonym",
        rules=[
            contains_the_word_w(w, features, n_words_per_example),
            contains_the_word_w(synonym, features_with_synonym, n_words_per_example),
        ],
    )


#############
# CONSTANTS #
#############

DEFAULT_RULE_FNS = tuple(
    sorted(
        {
            contains_the_word_w,
            contains_the_words_w_and_w_prime,
            contains_the_words_w_or_w_prime,
            does_not_contain_the_word_w,
            contains_the_words_w_or_w_prime_but_not_both,
            repeats_the_word_w,
            contains_the_word_w_between_k_and_k_prime_times,
            words_are_in_alphabetical_order,
            starts_with_the_word_w,
            word_w_appears_in_position_i,
            # starts_with_the_word_w_or_repeats_the_word_w_prime,  # hard (but maybe possible); passing examples rarely start with word w
            # starts_with_the_word_w_or_contains_the_word_w_prime_but_not_both,  # hard (but sometimes possible) to deduce from examples
            # starts_with_the_word_w_or_repeats_the_word_w_prime_but_not_both,  # hard (but maybe possible) to deduce from examples
            # starts_or_ends_with_the_word_w_but_not_both,  # hard to deduce from examples
            contains_the_word_w_followed_by_w_prime,
            contains_the_word_w_and_a_digit,
            contains_the_word_w_and_a_french_word,
            contains_the_word_w_and_is_in_alphabetical_order,
            contains_a_digit,
            contains_a_french_word,
            contains_the_word_w_at_least_k_times,
            contains_the_word_w_within_x_positions_of_the_word_w_prime,
            repeats_the_word_w_and_contains_a_french_word,
            repeats_the_word_w_and_is_in_alphabetical_order,
            repeats_the_word_w_and_contains_a_digit,
            repeats_the_word_w_and_starts_with_w,
            repeats_the_word_w_and_contains_the_word_w_prime,
            words_are_in_reverse_alphabetical_order,
            does_not_start_with_the_word_w,
            ends_with_the_word_w,
            # starts_with_the_word_w_or_contains_the_word_w_prime,  # passing examples almost never start with `w` (and when they do, the input also contains `w_prime`)
            starts_with_the_word_w_or_repeats_the_word_w_prime_but_not_both,
            ends_with_the_word_w_or_contains_the_word_w_prime,
            contains_the_word_w_and_the_word_w_prime_in_sorted_order,
        },
        key=lambda fn: fn.__name__,
    )
)

ROBUSTLY_LEARNT_RULE_FNS = tuple(
    sorted(
        {
            # contains_a_digit,
            # contains_the_word_w_followed_by_w_prime,
            starts_with_the_word_w,
            repeats_the_word_w_and_contains_the_word_w_prime,
            ends_with_the_word_w,
            does_not_start_with_the_word_w,
            contains_the_words_w_and_w_prime,
            does_not_contain_the_word_w,
            contains_the_word_w_at_least_k_times,
            contains_the_word_w_and_the_word_w_prime_in_sorted_order,
            contains_the_word_w,
            contains_the_word_w_and_a_digit,
        },
        key=lambda fn: fn.__name__,
    )
)
