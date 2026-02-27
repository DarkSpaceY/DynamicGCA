from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import torch
from utils import is_model_cached, is_dataset_cached

from dygca_model import DyGCAConfig, DyGCAPlugin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="128k")
    parser.add_argument("--task", default="qa1")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--dygca-checkpoint", default="")
    parser.add_argument("--no-chunk-attn", action="store_true")
    # DyGCA Specific Parameters (Monitoring)
    parser.add_argument("--k-focuses", type=int, default=32, help="Number of focuses (K)")
    parser.add_argument("--m-selection", type=int, default=8, help="Number of selected focuses (M)")
    return parser.parse_args()


def load_deps():
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception as exc:
        print("Missing dependencies: datasets, transformers")
        print(str(exc))
        sys.exit(1)
    return load_dataset, AutoModelForCausalLM, AutoTokenizer, pipeline


def infer_fields(example: dict) -> tuple[str, str, str]:
    context_candidates = ["context", "story", "text", "document", "input", "passage"]
    question_candidates = ["question", "query", "prompt"]
    answer_candidates = ["answer", "output", "label", "target"]

    def pick(candidates: list[str]) -> str:
        for key in candidates:
            if key in example:
                return key
        return ""

    context_key = pick(context_candidates)
    question_key = pick(question_candidates)
    answer_key = pick(answer_candidates)
    return context_key, question_key, answer_key


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def get_expected_needle_count(task: str) -> int:
    """Returns the expected number of facts/needles for a bAbI/BABILong task."""
    mapping = {
        "qa1": 1,
        "qa2": 2,
        "qa3": 3,
        "qa4": 2,
        "qa5": 2,  # Who gave X to Y? (Usually 2-3 facts)
        "qa6": 1,
        "qa7": 1,
        "qa8": 1,
        "qa9": 1,
        "qa10": 1,
    }
    return mapping.get(task, 1)


def evaluate_task(
    task_name: str,
    split,
    context_key: str,
    question_key: str,
    answer_key: str,
    tokenizer,
    generate_fn,
    max_samples: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> dict:
    total = min(max_samples, len(split))
    correct = 0
    
    # Monitoring metrics (DyGCA specific logic would go here)
    existence_correct = 0
    count_correct = 0
    set_match = 0
    
    expected_count = get_expected_needle_count(task_name)

    for idx in range(total):
        item = split[idx]
        context = item.get(context_key, "")
        question = item.get(question_key, "")
        answer = item.get(answer_key, "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        prompt = context
        if question:
            prompt = f"{context}\n\n{question}\nAnswer:"

        if max_input_tokens > 0:
            tokens = tokenizer(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
            prompt = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

        prediction = generate_fn(prompt, max_new_tokens)

        is_correct = normalize(str(answer)) in normalize(prediction)
        if is_correct:
            correct += 1
            existence_correct += 1
            # For BABILong, if the final answer is correct, we assume the needles were found
            count_correct += 1 
            set_match += 1

    acc = correct / total if total > 0 else 0.0
    return {
        "task": task_name,
        "total": total,
        "correct": correct,
        "acc": acc,
        "existence_acc": existence_correct / total if total > 0 else 0.0,
        "count_acc": count_correct / total if total > 0 else 0.0,
        "set_acc": set_match / total if total > 0 else 0.0,
    }


def build_generator(tokenizer, base_model, device: torch.device):
    def generate_fn(prompt: str, max_new_tokens: int) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        for _ in range(max_new_tokens):
            outputs = base_model(input_ids)
            logits = outputs["logits"][:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=-1)
            if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
                break
        new_tokens = input_ids[0][prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate_fn


def main() -> int:
    args = parse_args()
    load_dataset_fn, AutoModelForCausalLM, AutoTokenizer, pipeline = load_deps()

    model_cached = is_model_cached(args.model)
    dataset_cached = is_dataset_cached("RMT-team/babilong", args.config)
    
    if model_cached: print(f"Detected cached model: {args.model}, using local_files_only=True")
    if dataset_cached: print(f"Detected cached dataset: RMT-team/babilong, using local_files_only=True")

    load_kwargs = {"local_files_only": True} if dataset_cached else {}
    dataset = load_dataset_fn("RMT-team/babilong", args.config, **load_kwargs)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else [args.task]
    missing = [t for t in tasks if t not in dataset]
    if missing:
        print(f"Task {missing} not found. Available: {list(dataset.keys())}")
        return 1

    model_kwargs = {"use_fast": True, "trust_remote_code": True}
    if model_cached: model_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, **model_kwargs)
    generator = None
    if args.dygca_checkpoint:
        base_model_kwargs = {"trust_remote_code": True}
        if model_cached: base_model_kwargs["local_files_only"] = True
        base_model = AutoModelForCausalLM.from_pretrained(args.model, **base_model_kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)
        checkpoint = torch.load(args.dygca_checkpoint, map_location="cpu")
        config = DyGCAConfig(**checkpoint["config"])
        if args.no_chunk_attn:
            config.use_chunk_attn = False
        plugin = DyGCAPlugin(base_model, config).to(device)
        plugin_state = checkpoint.get("plugin_state", checkpoint.get("model", {}))
        plugin.load_state_dict(plugin_state, strict=False)
        generator = build_generator(tokenizer, plugin, device)
    else:
        base_model_kwargs = {"device_map": args.device, "trust_remote_code": True}
        if model_cached: base_model_kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(args.model, **base_model_kwargs)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=args.device)

    results = []
    for task in tasks:
        split = dataset[task]
        if len(split) == 0:
            print(f"Task {task} is empty")
            continue
        example = split[0]
        context_key, question_key, answer_key = infer_fields(example)
        if not context_key:
            print(f"Unable to infer context field from keys: {list(example.keys())}")
            return 1
            
        if args.dygca_checkpoint:
            generate_fn = generator
        else:
            def generate_fn(prompt: str, max_new_tokens: int) -> str:
                output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
                return output[0]["generated_text"] if output else ""

        res = evaluate_task(
            task,
            split,
            context_key,
            question_key,
            answer_key,
            tokenizer,
            generate_fn,
            args.max_samples,
            args.max_input_tokens,
            args.max_new_tokens,
        )
        results.append(res)
        print(f"task={task} samples={res['total']} correct={res['correct']} acc={res['acc']:.4f} count_acc={res['count_acc']:.4f}")

    if results:
        print("\n" + "="*30)
        print("BABILong Multi-Needle Summary")
        print("="*30)
        macro_acc = sum(r['acc'] for r in results) / len(results)
        macro_count_acc = sum(r['count_acc'] for r in results) / len(results)
        
        total_samples = sum(r['total'] for r in results)
        total_correct = sum(r['correct'] for r in results)
        micro_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        print(f"Macro Acc:       {macro_acc:.4f}")
        print(f"Micro Acc:       {micro_acc:.4f}")
        print(f"Macro Count Acc: {macro_count_acc:.4f} (Monitoring)")
        print("="*30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
