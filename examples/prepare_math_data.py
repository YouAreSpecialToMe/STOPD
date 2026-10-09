"""Mathematics data of the paper's main setting, in verl's parquet format.

  train: nvidia/Nemotron-Cascade-RL-Math (14,476 problems)
  val:   HuggingFaceH4/MATH-500

The data_source routes verl's rule-based math scorer (boxed answer, exact
match), used only for validation; training uses no task reward.
"""

import argparse
import os

import datasets

MATH_SCORER_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def to_verl(example, idx, split):
    return {
        "data_source": MATH_SCORER_SOURCE,
        "prompt": [{"role": "user", "content": example["problem"].strip() + " " + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(example["answer"])},
        "extra_info": {"split": split, "index": idx},
    }


def convert(dataset, split, path):
    dataset = dataset.map(lambda ex, i: to_verl(ex, i, split), with_indices=True,
                          remove_columns=dataset.column_names)
    dataset.to_parquet(path)
    print(f"{split}: {len(dataset)} -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="~/data/stopd_math")
    args = parser.parse_args()
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    convert(datasets.load_dataset("nvidia/Nemotron-Cascade-RL-Math", split="train"), "train",
            os.path.join(out, "train.parquet"))
    convert(datasets.load_dataset("HuggingFaceH4/MATH-500", split="test"), "test",
            os.path.join(out, "math500.parquet"))
