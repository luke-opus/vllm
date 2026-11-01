"""Summarize observed paired outputs without inventing a numerical tolerance."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    base, fixed = [
        json.loads((root / f"{name}.json").read_text()) for name in ("base", "fixed")
    ]
    assert base["parameters"] == fixed["parameters"]
    assert len(base["outputs"]) == len(fixed["outputs"]) == 2
    comparisons = []
    all_differences = []
    for before, after in zip(base["outputs"], fixed["outputs"]):
        assert before["prompt_token_ids"] == after["prompt_token_ids"]
        steps = []
        for index, (a, b) in enumerate(zip(before["logprobs"], after["logprobs"])):
            same_prefix = before["token_ids"][:index] == after["token_ids"][:index]
            common = sorted(set(a) & set(b), key=int)
            differences = [
                abs(a[token]["logprob"] - b[token]["logprob"]) for token in common
            ]
            top_a = {
                token
                for token, value in a.items()
                if value["rank"] is not None and value["rank"] <= 5
            }
            top_b = {
                token
                for token, value in b.items()
                if value["rank"] is not None and value["rank"] <= 5
            }
            if same_prefix:
                all_differences.extend(differences)
            steps.append(
                {
                    "step": index,
                    "same_generated_prefix": same_prefix,
                    "common_returned_tokens": common,
                    "common_returned_logprob_abs_differences": dict(
                        zip(common, differences)
                    ),
                    "max_abs_difference": max(differences) if differences else None,
                    "mean_abs_difference": sum(differences) / len(differences)
                    if differences
                    else None,
                    "top5_set_equal": top_a == top_b,
                    "base_top5": sorted(top_a, key=int),
                    "fixed_top5": sorted(top_b, key=int),
                    "all_returned_token_set_equal": set(a) == set(b),
                }
            )
        comparisons.append(
            {
                "prompt": before["prompt"],
                "base_tokens": before["token_ids"],
                "fixed_tokens": after["token_ids"],
                "base_text": before["text"],
                "fixed_text": after["text"],
                "matching_tokens": sum(
                    x == y for x, y in zip(before["token_ids"], after["token_ids"])
                ),
                "steps": steps,
            }
        )
    result = {
        "scope": "Descriptive statistics on this paired run; no accuracy improvement or numerical equivalence claim",
        "same_prefix_common_returned_logprob_values": len(all_differences),
        "same_prefix_max_abs_logprob_difference": max(all_differences),
        "same_prefix_mean_abs_logprob_difference": sum(all_differences)
        / len(all_differences),
        "steps_with_top5_set_changes": sum(
            not step["top5_set_equal"] for item in comparisons for step in item["steps"]
        ),
        "comparisons": comparisons,
    }
    (root / "comparison-details.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "comparisons"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
