# Overview

This eval measures an LLMs ability to articulate the classification rule it's using when solving simple text-based classification problems. Specifically, it contains two categories of tasks:

| Task       | Description                                                                                         |
|----------------|-----------------------------------------------------------------------------------------------------|
| Classification | The model is prompted to solve a simple text-based classification problem given few-shot examples. |
| Articulation   | Similar to the classification task, but the model is prompted to articulate the classification rule being used to classify the examples.              |

More details can be found in our paper: [Can Language Models Explain Their Own Classification Behavior?](www.broken-link.com)

# Setup

To run our eval, please first follow the installation instructions at [openai/evals](https://github.com/openai/evals).

Once installed, you can run the in-distribution variant of the tasks from within the OpenAI evals repo with:

```bash
oaieval <model> articulate-rules.<task>.in-distribution.<rule>
```

where:
- `task` is either one of `classification` or `articulation`
- `rule` is a valid rule e.g. `contains_a_digit`

Similarly, all out-of-distribution variants can be run with:

```bash
oaieval <model> articulate-rules.<task>.out-of-distribution.<rule>.<attack>
```

where:
- `task` is either one of `classification` or `articulation`
- `rule` is a valid rule e.g. `contains_a_digit`
- `attack` is a valid attack e.g. `...`

# Customiziation

We've provided a file `generate-eval.py` which allows you to create custom variants of our eval to e.g. increase the number of few-shot examples. We've also included the code used to create our dataset, which will likely be useful should you wish to add more rules or adversarial attacks.
