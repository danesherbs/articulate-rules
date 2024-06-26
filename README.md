# Overview

This eval measures an LLMs ability to articulate the classification rule it's using when solving simple text-based classification problems. Specifically, it contains two categories of tasks:

| Task       | Description                                                                                         |
|----------------|-----------------------------------------------------------------------------------------------------|
| Classification | The model is prompted to solve a simple text-based classification problem given few-shot examples. |
| Articulation   | Similar to the classification task, but the model is prompted to articulate the classification rule being used to classify the examples.              |

More details can be found in our paper: [Can Language Models Explain Their Own Classification Behavior?](www.broken-link.com)

# Setup

To run our eval, please clone the OpenAI evals [repo](https://github.com/openai/evals) and follow the installation instructions.

Within the OpenAI evals repo, you can run the in-distribution variant of Articulate Rules with:

```bash
oaieval <model> articulate-rules.<task>.in-distribution.<rule>
```

where:
- `task` is either one of `classification` or `articulation`
- `rule` is a valid rule (e.g. `contains_a_digit`)

Similarly, the out-of-distribution variants can be run with:

```bash
oaieval <model> articulate-rules.<task>.out-of-distribution.<rule>.<attack>
```

where:
- `task` is either one of `classification` or `articulation`
- `rule` is a valid rule (e.g. `contains_a_digit`)
- `attack` is a valid adversarial attack (e.g. `SynonymAttackBehaviour`)

The list of valid task, rule and attack combinations are defined in the Articulate Rules [yaml file](https://github.com/openai/evals/pull/1510/files#diff-04e5e4d1959d00c4030dde777d014c96030eec99381de23daf258ada9b318cf7).

# Customiziation

We've provided two scripts, `scripts/create-eval.py` and `scripts/create-eval-spec.py`, which allow you to create custom variants of our eval in the format of an OpenAI eval. This may be useful if you wish to increase the number of few-shot examples, for example. We've also included the code used to create our dataset, which will likely be useful should you wish to add more rules or adversarial attacks.
