import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize GRPO debate inputs into single-paragraph outputs (keeps \n and <> structures)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../model/Qwen2.5-7B-Instruct",
        help="Path to model or HF model name (base model, no LoRA)",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.path.join("../data", "GRPO.json"),
        help="Path to GRPO JSON file (list of {input, output})",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join("../data", "GRPO_summaries.json"),
        help="Path to save summarized JSON (list of {input, output})",
    )
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum generation length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (lower = more deterministic)"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading base model (no LoRA/QLoRA): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def build_prompt(input_text: str) -> str:
    # 中文任务说明：生成单段摘要，保留输入中出现的结构字符，不使用列表或编号
    instruction = (
        "你是辩论助手。请对下方给定的输入进行中文摘要：\n"
        "- 输出为一段完整、连贯的段落，不使用列表或编号；\n"
        "- 对每一段内容都要覆盖核心立场、关键论据与逻辑脉络，语言自然简洁；（包括两方的立论和上一轮发言）\n"
        "- 不要添加标题、代码块或额外标记；\n"
        "- 保留原文出现的结构字符，如\\n与尖括号<>中的内容，不要移除、替换或转义；\n"
        "- 仅输出摘要正文。"
    )
    prompt = f"{instruction}\n\n【输入】\n{input_text}\n\n【摘要】"
    return prompt


def generate_response(model, tokenizer, prompt: str, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Return only the generated suffix after prompt
    if response.startswith(prompt):
        return response[len(prompt) :].strip()
    return response.strip()


def read_grpo_inputs(input_path: str) -> List[str]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects with 'input' fields.")

    inputs: List[str] = []
    for i, item in enumerate(data):
        txt = item.get("input")
        if not isinstance(txt, str) or not txt.strip():
            # Skip invalid entries but print a message for traceability
            print(f"[WARN] Skipping item {i}: missing/empty 'input'")
            continue
        inputs.append(txt)
    print(f"Collected {len(inputs)} inputs from {len(data)} records.")
    return inputs


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    inputs = read_grpo_inputs(args.input_path)

    results: List[Dict[str, str]] = []
    for idx, inp in enumerate(inputs, start=1):
        print(f"\n===== ITEM {idx}/{len(inputs)} =====")
        prompt = build_prompt(inp)
        summary = generate_response(
            model,
            tokenizer,
            prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Emit as {input, output} to keep GRPO-compatible schema
        item = {
            "input": inp,
            "output": summary,
        }
        results.append(item)

        print("SUMMARY OUTPUT:")
        print(summary)

    out_dir = os.path.dirname(args.output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} items to: {args.output_path}")


if __name__ == "__main__":
    main()