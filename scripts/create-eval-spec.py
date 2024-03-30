import yaml
from rules import DEFAULT_RULE_FNS
from attacks import DEFAULT_ATTACK_BEHAVIOUR_CLASSES


rules = [rule_fn.__name__ for rule_fn in DEFAULT_RULE_FNS]
attacks = [attack.__name__ for attack in DEFAULT_ATTACK_BEHAVIOUR_CLASSES]

yaml_dict = {
    "articulate-rules": {
        "id": "articulate-rules.classification.in-context.in-distribution.v0",
        "description": "Evaluates how well a model can articulate their internal process when solving simple classification problems",
        "metrics": ["accuracy", "boostrap_std"],
    }
}

for rule in rules:
    key = f"articulate-rules.classification.in-context.in-distribution.{rule}.v0"

    yaml_dict[key] = {
        "class": "evals.elsuite.basic.fuzzy_match:FuzzyMatch",
        "args": {
            "samples_jsonl": f"articulate_rules/classification/in-context/in-distribution/{rule}.jsonl"
        },
    }

for rule in rules:
    for attack in attacks:
        key = f"articulate-rules.classification.in-context.out-of-distribution.{rule}.{attack}.v0"

        yaml_dict[key] = {
            "class": "evals.elsuite.basic.fuzzy_match:FuzzyMatch",
            "args": {
                "samples_jsonl": f"articulate_rules/classification/in-context/out-of-distribution/{rule}/{attack}.jsonl"
            },
        }

for rule in rules:
    key = f"articulate-rules.articulation.in-context.multiple-choice.{rule}.v0"

    yaml_dict[key] = {
        "class": "evals.elsuite.basic.includes:Includes",
        "args": {
            "samples_jsonl": f"articulate_rules/articulation/in-context/multiple-choice/{rule}.jsonl"
        },
    }

for rule in rules:
    key = f"articulate-rules.articulation.in-context.freeform.{rule}.v0"

    yaml_dict[key] = {
        "class": "evals.elsuite.modelgraded.classify:ModelBasedClassify",
        "args": {
            "samples_jsonl": f"articulate_rules/articulation/in-context/freeform/{rule}.jsonl",
            "eval_type": "cot_classify",
            "modelgraded_spec": "fact",
        },
    }


yaml_str = yaml.dump(yaml_dict, default_flow_style=False)

with open("articulate-rules.yaml", "w") as file:
    file.write(yaml_str)
