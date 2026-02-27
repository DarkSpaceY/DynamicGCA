from __future__ import annotations

import argparse
import os
import math
import random
from functools import partial
from typing import Iterable, Iterator

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

from dygca_model import DyGCAConfig, DyGCAPlugin


class BabiDataset(Dataset):
    """bAbI 专用数据集：支持 Qwen Instruct 对话模板、Label Masking 与 Attention Mask"""
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 'Muennighoff/babi' 字段通常是 context, question, target 或 story, question, answer
        story = item.get("story") or item.get("passage") or item.get("context", "")
        question = item.get("question", "")
        answer = item.get("answer") or item.get("target", "")
        
        # 使用 Instruct 模板构建对话
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided story."},
            {"role": "user", "content": f"Story: {story}\nQuestion: {question}"},
            {"role": "assistant", "content": str(answer)}
        ]
        
        # 1. Tokenization (显式指定 tokenize=True)
        def to_list(ids_or_str):
            if isinstance(ids_or_str, str):
                return self.tokenizer.encode(ids_or_str, add_special_tokens=False)
            if torch.is_tensor(ids_or_str):
                return ids_or_str.detach().cpu().tolist()
            # 处理 nested list 的情况 (某些 tokenizer 可能返回 [[ids]])
            if isinstance(ids_or_str, (list, tuple)) and len(ids_or_str) > 0 and isinstance(ids_or_str[0], (list, tuple)):
                return list(ids_or_str[0])
            return list(ids_or_str)

        res = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        full_ids = to_list(res)
        
        # 2. 构造 Prompt 部分以确定 Mask 边界
        prompt_messages = messages[:-1]
        p_res = self.tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
        prompt_ids = to_list(p_res)
        
        # 3. 转换为 Tensor 并进行类型检查
        try:
            full_ids_tensor = torch.tensor(full_ids, dtype=torch.long)
        except Exception as e:
            # 最后的防线：如果还是失败，尝试强制转换每个元素
            try:
                full_ids_tensor = torch.tensor([int(x) for x in full_ids], dtype=torch.long)
            except:
                print(f"Failed to convert full_ids: {full_ids}")
                raise e

        labels = full_ids_tensor.clone()
        
        # 3. Label Masking
        prompt_len = len(prompt_ids)
        labels[:prompt_len] = -100
        
        return {
            "input_ids": full_ids_tensor,
            "labels": labels,
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long)
        }

def babi_collate_fn(batch, tokenizer):
    """处理不定长样本的 Padding 并生成 attention_mask"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    
    max_len = max(x.size(0) for x in input_ids)
    
    padded_inputs = []
    padded_labels = []
    padded_masks = []
    
    for ids, lbls, mask in zip(input_ids, labels, attention_masks):
        pad_len = max_len - ids.size(0)
        pad_id = tokenizer.pad_token_id
        # 填充 input_ids (使用 pad_token_id)
        padded_inputs.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        # 填充 labels (-100)
        padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)]))
        # 填充 attention_mask (0)
        padded_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
        
    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(padded_masks)
    }

class TokenStreamDataset(Dataset):
    """通用数据集：按样本产出，不强制 seq_len 分段"""
    def __init__(self, token_iter: Iterable[list[int]]):
        super().__init__()
        # 如果是流式迭代器，且数据集很大，这里会有内存风险。
        # 但目前我们主要针对 bAbI 优化。
        self.samples = []
        for ids in token_iter:
            self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long)
        }

class StreamingTokenDataset(IterableDataset):
    """真正的流式数据集：按样本产出，不强制 seq_len 分段"""
    def __init__(self, token_iter: Iterable[list[int]]):
        super().__init__()
        self.token_iter = token_iter

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for ids in self.token_iter:
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(ids, dtype=torch.long)
            }

def general_collate_fn(batch, tokenizer):
    """通用 Padding 处理"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    max_len = max(x.size(0) for x in input_ids)
    
    padded_inputs = []
    padded_labels = []
    for ids, lbls in zip(input_ids, labels):
        pad_len = max_len - ids.size(0)
        pad_id = tokenizer.pad_token_id
        padded_inputs.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(padded_inputs),
        "labels": torch.stack(padded_labels)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Muennighoff/babi")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of train data to use for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--k-focuses", type=int, default=32)
    parser.add_argument("--m-selection", type=int, default=8)
    parser.add_argument("--distribution", type=str, default="beta", choices=["beta", "gaussian", "laplace", "studentt"])
    parser.add_argument("--gate-bias-init", type=float, default=-2.0, help="Initial bias for the gating mechanism")
    parser.add_argument("--diversity-lambda", type=float, default=0.1)
    parser.add_argument("--max-position-embeddings", type=int, default=32768)
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument("--no-chunk-attn", action="store_true")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--target-acc", type=float, default=0.95, help="Stop training when sliding window accuracy reaches this value")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Number of steps for gradient accumulation")
    return parser.parse_args()


def extract_text_fields(sample: dict) -> list[str]:
    candidates = ["text", "content", "document", "input", "passage"]
    texts = []
    for key in candidates:
        if key in sample and isinstance(sample[key], str):
            texts.append(sample[key])
    return texts


def text_iterator(dataset, text_field: str) -> Iterable[str]:
    for sample in dataset:
        if text_field and text_field in sample and isinstance(sample[text_field], str):
            yield sample[text_field]
            continue
        fields = extract_text_fields(sample)
        if fields:
            yield fields[0]


def token_iterator(text_iter: Iterable[str], tokenizer) -> Iterable[list[int]]:
    for text in text_iter:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            ids = ids + [tokenizer.eos_token_id]
        yield ids


def make_dataloader(token_iter: Iterable[list[int]], seq_len: int, batch_size: int) -> DataLoader:
    dataset = TokenStreamDataset(token_iter, seq_len)
    return DataLoader(dataset, batch_size=batch_size)


def train_step(model: DyGCAPlugin, batch: dict[str, torch.Tensor], device: torch.device, grad_accum_steps: int = 1) -> dict:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    # 获取并透传 attention_mask
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    
    # 增加 Token 准确率计算 (仅针对 labels != -100 的部分)
    logits = outputs["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != -100)
    preds = shift_logits.argmax(dim=-1)
    correct = (preds == shift_labels) & mask
    accuracy = correct.sum().float() / (mask.sum().float() + 1e-9)
    
    loss = outputs["loss"] / grad_accum_steps
    loss.backward()
    
    return {
        "loss": float(outputs["loss"].item()),
        "lm_loss": float(outputs["lm_loss"].item()),
        "diversity_loss": float(outputs["diversity_loss"].item()),
        "accuracy": float(accuracy.item()),
        "correct_tokens": int(correct.sum().item()),
        "total_tokens": int(mask.sum().item()),
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 针对 macOS MPS
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_step(model: DyGCAPlugin, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # 这里的 logits 是 (bsz, seq_len, vocab_size)
            # 我们需要比较 assistant 回答部分的 EM (Exact Match)
            # labels 中非 -100 的部分即为答案
            
            bsz = input_ids.size(0)
            for i in range(bsz):
                # 提取该样本的答案 token 位置
                sample_labels = labels[i]
                answer_mask = (sample_labels != -100)
                
                if not answer_mask.any():
                    continue
                
                # 获取预测结果
                # 注意：在 Causal LM 中，位置 t 的预测在 logits[t-1] 处
                sample_logits = logits[i]
                # 移位以对齐预测与标签
                shift_logits = sample_logits[:-1, :]
                shift_labels = sample_labels[1:]
                shift_answer_mask = answer_mask[1:]
                
                if not shift_answer_mask.any():
                    continue
                
                preds = shift_logits.argmax(dim=-1)
                
                # 提取答案部分的预测和真实标签
                ans_preds = preds[shift_answer_mask]
                ans_labels = shift_labels[shift_answer_mask]
                
                # Exact Match: 整个答案序列必须完全一致
                if torch.equal(ans_preds, ans_labels):
                    total_correct += 1
                total_samples += 1
    
    model.train()
    return total_correct / (total_samples + 1e-9)


def is_model_cached(model_name: str) -> bool:
    """Check if the model is available in the local Hugging Face cache."""
    # If it's a local path that exists, return True
    if os.path.isdir(model_name):
        return True
    
    # Try to find a key file (like config.json) in the cache
    try:
        # Use try_to_load_from_cache to check for the presence of config.json
        filepath = try_to_load_from_cache(model_name, "config.json")
        if filepath is not None and filepath != _CACHED_NO_EXIST:
            return True
    except Exception:
        pass
    return False


def is_dataset_cached(dataset_name: str, config: str = None) -> bool:
    """Check if the dataset is available in the local cache."""
    if os.path.isdir(dataset_name):
        return True
    try:
        # Attempt to load metadata/config with local_files_only
        load_dataset(dataset_name, config, local_files_only=True)
        return True
    except:
        return False


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    
    # 0. 设备检测提前：支持 CUDA -> MPS -> CPU
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif args.device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 检测模型和数据集是否已缓存
    model_cached = is_model_cached(args.base_model)
    dataset_cached = is_dataset_cached(args.dataset, args.config)
    
    if model_cached:
        print(f"Detected cached model: {args.base_model}, using local_files_only=True")
    if dataset_cached:
        print(f"Detected cached dataset: {args.dataset}, using local_files_only=True")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True, local_files_only=model_cached)
    
    # 3. Qwen 特有 EOS/PAD 处理
    if "qwen" in args.base_model.lower():
        # Qwen 2.5 使用 <|endoftext|> 作为 EOS，其 ID 为 151643
        # 使用官方推荐方式赋值，避免硬编码字符串风险
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if "babi" in args.dataset.lower():
        print(f"Loading bAbI dataset: {args.dataset}")
        # 从 train split 中划分 val，避免 test 数据泄露
        full_dataset = load_dataset(args.dataset, split=args.split, local_files_only=dataset_cached)
        
        # 仅保留事实类任务 (Fact-based tasks: 1, 2, 3, 6, 10, 11, 12, 13)
        fact_tasks = {1, 2, 3, 6, 10, 11, 12, 13}
        def is_fact_task(example):
            # 假设 dataset 中有 'task_id' 字段，或者可以从 story/question 结构中推断
            # 注意：HuggingFace bAbI dataset 的结构中，每个样本通常对应一个 task
            # 如果加载的是 'facebook/babi'，通常 split 已经是按 task 分好的
            # 如果是合并的数据集，我们需要过滤。
            if 'task_id' in example:
                return example['task'] in fact_tasks
            # 如果没有直接的 task_id，可以尝试从 dataset 名字中获取 (如果是按 task 加载的)
            return True # 默认全选，除非能明确过滤

        # 尝试根据数据集名称或字段过滤
        if 'task_id' in full_dataset.column_names:
            full_dataset = full_dataset.filter(lambda x: x['task_id'] in fact_tasks)
            print(f"Filtered for fact-based tasks. Remaining: {len(full_dataset)}")
        
        # 固定种子进行划分
        split_data = full_dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        
        train_data = split_data["train"]
        val_data = split_data["test"]
        
        print(f"bAbI dataset split: train={len(train_data)}, val={len(val_data)}")
        
        train_dataset = BabiDataset(train_data, tokenizer)
        val_dataset = BabiDataset(val_data, tokenizer)
        
        dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=partial(babi_collate_fn, tokenizer=tokenizer)
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=partial(babi_collate_fn, tokenizer=tokenizer)
        )
    else:
        print(f"Loading dataset: {args.dataset} (config={args.config})")
        dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=not args.no_streaming)
        texts = text_iterator(dataset, args.text_field)
        tokens = token_iterator(texts, tokenizer)
        # 2. 通用模式也移除强制 seq_len 切分，改为样本产出
        if not args.no_streaming:
            train_dataset = StreamingTokenDataset(tokens)
            dataloader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                collate_fn=partial(general_collate_fn, tokenizer=tokenizer)
            )
        else:
            train_dataset = TokenStreamDataset(tokens)
            dataloader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                collate_fn=partial(general_collate_fn, tokenizer=tokenizer)
            )
        # 非 bAbI 数据集暂时不启用验证集早停，或者可以根据需要后续添加
        val_dataloader = None

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32, 
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=model_cached
    ).to(device)

    # 1. Apply LoRA to base model (bAbI fine-tuning strategy)
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)
        print("LoRA applied to base model.")
        # 卸载不必要的权重或转换为更节省内存的格式（如果可能）
        # 这里我们保持 fp32 因为是 CPU 训练，但可以确保没有重复加载

    if not args.train_base:
        for param in base_model.parameters():
            param.requires_grad = False
    vocab_size = base_model.config.vocab_size if hasattr(base_model.config, "vocab_size") else len(tokenizer)

    # 2. Setup DyGCA Plugin
    config = DyGCAConfig(
        vocab_size=vocab_size,
        num_heads=args.num_heads,
        hidden_size=base_model.config.hidden_size,
        max_position_embeddings=args.max_position_embeddings,
        k_focuses=args.k_focuses,
        m_selection=args.m_selection,
        distribution=args.distribution,
        gate_bias_init=args.gate_bias_init,
        diversity_lambda=args.diversity_lambda,
        use_chunk_attn=not args.no_chunk_attn,
        use_gate=not args.no_gate,
    )
    model = DyGCAPlugin(base_model, config)
    model.to(device)
    # Ensure plugin and LoRA params are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    # 3. Training Loop
    model.train()
    
    print(f"Starting training for {args.epochs} epochs (Effective Batch Size: {args.batch_size * args.grad_accum_steps})...")
    
    early_stop = False
    global_step = 0
    
    for epoch in range(args.epochs):
        # 使用 tqdm 进度条显示当前 Epoch
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        accum_loss = 0.0
        accum_accuracy = 0.0
        
        for micro_step, batch in enumerate(pbar):
            metrics = train_step(model, batch, device, grad_accum_steps=args.grad_accum_steps)
            
            accum_loss += metrics['loss']
            accum_accuracy += metrics['accuracy']
            
            # 梯度更新逻辑 (每个有效 Batch)
            if (micro_step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                # 只有在权重更新时才计算平均训练指标
                avg_loss = accum_loss / args.grad_accum_steps
                avg_acc = accum_accuracy / args.grad_accum_steps
                
                # 更新进度条信息
                metrics_to_show = {
                    "loss": f"{avg_loss:.4f}",
                    "train_acc": f"{avg_acc:.4f}",
                }
                pbar.set_postfix(metrics_to_show)
                
                accum_loss = 0.0
                accum_accuracy = 0.0

            elif (micro_step + 1) % 10 == 0:
                # 中间微步仅更新 postfix
                pbar.set_postfix({"micro_loss": f"{metrics['loss']:.4f}"})
        
        # Epoch 结束时进行一次完整验证
        val_acc = 0.0
        if val_dataloader is not None:
            tqdm.write(f"\n[Epoch {epoch+1}] Running full validation...")
            val_acc = validate_step(model, val_dataloader, device)
            tqdm.write(f"[Epoch {epoch+1}] Validation Accuracy: {val_acc:.4f}")
            
            # 检查是否达到目标准确度
            if val_acc >= args.target_acc:
                tqdm.write(f"[Target Reached] Validation accuracy {val_acc:.4f} reached target {args.target_acc}!")
                early_stop = True
        
        # 如果已经达到目标，则停止训练
        if early_stop:
            tqdm.write(f"Training stopped after Epoch {epoch+1} as target was reached.")
            break
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        # 保存 DyGCA Config
        torch.save(config, os.path.join(args.save_dir, "dygca_config.pt"))
        
        # 保存所有可训练参数 (包含 LoRA 和 DyGCA Plugin)
        trainable_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
        torch.save(trainable_state_dict, os.path.join(args.save_dir, "trainable_params.pt"))
        
        # 推荐：使用 PEFT 接口保存 LoRA 部分
        if args.use_lora:
            model.base_model.save_pretrained(os.path.join(args.save_dir, "lora"))
            
        print(f"Model saved to {args.save_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
