from __future__ import annotations

import argparse
import os
import math
from typing import Iterable, Iterator

import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from dygca_model import DyGCAConfig, DyGCAPlugin


class BabiDataset(Dataset):
    """bAbI 专用数据集：支持 Qwen Instruct 对话模板、Label Masking 与 Attention Mask"""
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        for item in dataset:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Story:\n{item.get('passage', '')}\nQuestion:\n{item.get('question', '')}"},
                {"role": "assistant", "content": item.get("answer", "")}
            ]
            
            # 1. 使用 apply_chat_template 构造完整序列 (含答案和 EOS)
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            
            # 2. 构造 Prompt 部分（含 Assistant 起始标记，但不含答案）
            prompt_messages = messages[:-1]
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            
            # 3. 安全的 Label Masking：对齐 Prompt 和 Full 序列
            # 通过 prompt_ids 的长度来确定掩码边界，避免 BPE 分词合并导致的问题
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            
            self.samples.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def babi_collate_fn(batch):
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
        # 填充 input_ids (使用 151643/pad_token_id)
        padded_inputs.append(torch.cat([ids, torch.full((pad_len,), 151643, dtype=torch.long)]))
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

def general_collate_fn(batch):
    """通用 Padding 处理"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    max_len = max(x.size(0) for x in input_ids)
    
    padded_inputs = []
    padded_labels = []
    for ids, lbls in zip(input_ids, labels):
        pad_len = max_len - ids.size(0)
        padded_inputs.append(torch.cat([ids, torch.full((pad_len,), 151643, dtype=torch.long)]))
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
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--k-focuses", type=int, default=32)
    parser.add_argument("--m-selection", type=int, default=8)
    parser.add_argument("--distribution", default="beta")
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


def main() -> int:
    args = parse_args()
    
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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    
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
        # 4. 弃用 Streaming，回归标准 Dataset，支持 Shuffle
        dataset = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
        train_dataset = BabiDataset(dataset, tokenizer)
        dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=babi_collate_fn)
    else:
        print(f"Loading dataset: {args.dataset} (config={args.config})")
        dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=not args.no_streaming)
        texts = text_iterator(dataset, args.text_field)
        tokens = token_iterator(texts, tokenizer)
        # 2. 通用模式也移除强制 seq_len 切分，改为样本产出
        if not args.no_streaming:
            train_dataset = StreamingTokenDataset(tokens)
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=general_collate_fn)
        else:
            train_dataset = TokenStreamDataset(tokens)
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=general_collate_fn)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

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
    data_iter = iter(dataloader)
    
    total_correct = 0
    total_tokens = 0
    epoch_samples = 0
    accum_loss = 0.0
    accum_accuracy = 0.0
    
    print(f"Starting training for {args.max_steps} steps (Effective Batch Size: {args.batch_size * args.grad_accum_steps})...")
    
    # 使用 tqdm 进度条
    pbar = tqdm(range(args.max_steps), desc="Training")
    
    for step in pbar:
        batch = next(data_iter, None)
        if batch is None:
            # Epoch 结束统计
            global_acc = total_correct / (total_tokens + 1e-9)
            tqdm.write(f"\n[Epoch End] Global Accuracy: {global_acc:.4f}")
            
            if global_acc >= args.target_acc:
                tqdm.write(f"Target global accuracy {args.target_acc} reached. Stopping training.")
                break
                
            total_correct = 0
            total_tokens = 0
            epoch_samples = 0
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        metrics = train_step(model, batch, device, grad_accum_steps=args.grad_accum_steps)
        
        # 累积全局统计
        total_correct += metrics['correct_tokens']
        total_tokens += metrics['total_tokens']
        epoch_samples += len(batch['input_ids'])
        
        accum_loss += metrics['loss']
        accum_accuracy += metrics['accuracy']
        
        # 梯度更新逻辑
        if (step + 1) % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # 只有在权重更新时才计算平均指标
            avg_loss = accum_loss / args.grad_accum_steps
            avg_acc = accum_accuracy / args.grad_accum_steps
            
            # 更新进度条信息
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc": f"{avg_acc:.4f}",
                "global_acc": f"{total_correct/(total_tokens+1e-9):.4f}"
            })
            
            accum_loss = 0.0
            accum_accuracy = 0.0

        if (step + 1) % 10 == 0 and (step + 1) % args.grad_accum_steps != 0:
            # 中间步骤仅更新 postfix，不打印新行以保持进度条整洁
            pbar.set_postfix({"step_loss": f"{metrics['loss']:.4f}", "step_acc": f"{metrics['accuracy']:.4f}"})
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "dygca.pt")
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("base_model.")}
        torch.save(
            {
                "base_model": args.base_model,
                "plugin_state": state,
                "config": config.__dict__,
            },
            save_path,
        )
        print(f"Saved: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
