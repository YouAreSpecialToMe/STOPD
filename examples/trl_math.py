"""STOPD with TRL's DistillationTrainer on the mathematics data of examples/prepare_math_data.py.

    python examples/trl_math.py --data ~/data/stopd_math/train.parquet --output_dir out/trl-stopd \\
        --student Qwen/Qwen3-1.7B-Base --teacher Qwen/Qwen3-4B-Instruct-2507 --use_vllm

Use `accelerate launch` for several GPUs. With --use_vllm, generation runs in colocate mode (each process holds its
own vLLM engine), which is where per-rollout budgets save generation time. Arguments not listed below keep TRL's
defaults; edit the STOPDConfig call to change them.
"""

import argparse

import pandas as pd
from datasets import Dataset

from stopd.integrations.trl import STOPDConfig, STOPDTrainer


def load_prompts(path, limit=None):
    """The conversations of verl's parquet; TRL applies the student's chat template to them, as verl does."""
    frame = pd.read_parquet(path)
    prompts = [[{"role": m["role"], "content": m["content"]} for m in row] for row in frame["prompt"]]
    return Dataset.from_dict({"prompt": prompts[:limit] if limit else prompts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="train.parquet from examples/prepare_math_data.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_len", type=int, default=16384, help="L, the max completion length")
    parser.add_argument("--block", type=int, default=128, help="K")
    parser.add_argument("--warmup_steps", type=int, default=25, help="initial Vanilla OPD steps")
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=8, help="prompts per device per micro-batch")
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    args = parser.parse_args()

    config = STOPDConfig(
        output_dir=args.output_dir,
        max_completion_length=args.max_len,
        stopd_block=args.block,
        stopd_warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        beta=1.0,                       # reverse KL, as in the paper
        temperature=1.0,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=args.save_steps,
        logging_steps=1,
        report_to="none",
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    trainer = STOPDTrainer(model=args.student, teacher_model=args.teacher, args=config,
                           train_dataset=load_prompts(args.data, args.limit))
    trainer.train()


if __name__ == "__main__":
    main()
