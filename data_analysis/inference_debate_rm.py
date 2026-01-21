import argparse
import json
import os

import torch
import torch.nn as nn
from jinja2 import Environment, FileSystemLoader
from peft import PeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SaMerClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.scoring_layer = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(2048, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scoring_layer(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Score GRPO outputs using Debate RM and save to JSON")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model or HF model name")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to saved checkpoint directory")
    parser.add_argument("--template_path", type=str, required=True, help="Path to Jinja template file")
    parser.add_argument("--example_path", type=str, required=True, help="Path to GRPO examples JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save scored JSON outputs")
    parser.add_argument("--device_ids", type=str, default="", help="Comma-separated GPU IDs, e.g. '0,1,2,3'")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for batched inference")
    parser.add_argument("--store_ground_truth", action="store_true", default=True)
    parser.add_argument("--disable_match", action="store_true", default=True)
    return parser.parse_args()


def load_model_and_tokenizer(args):
    print(f"Loading base model (sequence classification): {args.model_path}")
    
    # Check if we are in "Scalar Custom Head" mode (implied by missing rm_head.pt)
    rm_head_path = os.path.join(args.adapter_path, "rm_head.pt")
    is_scalar_custom = not os.path.exists(rm_head_path)
    
    if is_scalar_custom:
        print("rm_head.pt not found. Assuming Scalar Custom Head mode (1 output dim, no aggregator).")
        num_labels = 1
    else:
        print("rm_head.pt found. Assuming Multi-dim Custom Head mode (8 dims + aggregator).")
        # Pre-load head state to check num_labels if possible, or default to 8
        try:
            head_state = torch.load(rm_head_path, map_location="cpu")
            num_labels = int(head_state.get("num_labels", 8))
        except Exception:
            num_labels = 8

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Get hidden size for custom head
    hidden_size = getattr(base_model.config, "hidden_size", None)
    if hidden_size is None:
        sizes = getattr(base_model.config, "hidden_sizes", [])
        hidden_size = sizes[0] if sizes else None
    if hidden_size is None:
        raise RuntimeError("无法从模型配置中推断 hidden_size，无法构建 RM 头部。")

    # If Scalar Custom Head mode, we must swap the head BEFORE loading adapter
    # so that PeftModel loads the weights into the correct structure.
    if is_scalar_custom:
        head = SaMerClassifier(hidden_size, output_dim=1)
        # Identify and replace score head
        if hasattr(base_model, "score"):
            base_model.score = head
        elif hasattr(base_model, "classifier"):
            base_model.classifier = head
        elif hasattr(base_model, "scores"):
            base_model.scores = head
        else:
            # Fallback, try to set 'score' and hope it's used
            base_model.score = head
        
        aggregator = None
        index_to_dim = None
        ordered_dims = None
        dim_to_index = None
    
    # Load Adapter
    adapter_stub = os.path.join(args.adapter_path, "adapter_model")
    has_adapter = os.path.exists(adapter_stub + ".safetensors") or os.path.exists(adapter_stub + ".bin") or os.path.exists(os.path.join(args.adapter_path, "adapter_model.safetensors")) or os.path.exists(os.path.join(args.adapter_path, "adapter_model.bin"))
    
    if has_adapter:
        print(f"Loading adapter from: {args.adapter_path}")
        model = PeftModelForSequenceClassification.from_pretrained(base_model, args.adapter_path)
    else:
        pt_bin = os.path.join(args.adapter_path, "pytorch_model.bin")
        if os.path.exists(pt_bin):
            print(f"Found state dict at {pt_bin}; loading into base model (strict=False)")
            state = torch.load(pt_bin, map_location="cpu")
            base_model.load_state_dict(state, strict=False)
            model = base_model
        else:
            print(f"No adapter/state dict found at {args.adapter_path}; using base model")
            model = base_model

    # If Multi-dim Custom Head mode, we load head and aggregator from rm_head.pt
    if not is_scalar_custom:
        head_state = torch.load(rm_head_path, map_location="cpu")
        # num_labels already set above
        
        head = SaMerClassifier(hidden_size, num_labels)
        aggregator = nn.Linear(num_labels, 1, bias=False)

        head.load_state_dict(head_state["head"], strict=True)
        aggregator.load_state_dict(head_state["aggregator"], strict=True)

        # Replace head in the wrapped model
        try:
            model.base_model.score = head
        except AttributeError:
            if hasattr(model.base_model, "classifier"):
                model.base_model.classifier = head
            elif hasattr(model.base_model, "scores"):
                model.base_model.scores = head
            else:
                model.base_model.score = head

        # Setup dimension mappings
        ordered_dims_match = ["逻辑性","吸引力","针对性","清晰度","有效性","灵活性","原创性","流畅性"]
        ordered_dims_default = ["流畅性","原创性","灵活性","逻辑性","针对性","有效性","清晰度","吸引力"]
        
        if args.disable_match:
            ordered_dims = ordered_dims_default
            index_to_dim = {i: ordered_dims_default[i] for i in range(len(ordered_dims_default))}
            dim_to_index = {dim: i for i, dim in enumerate(ordered_dims_default)}
        else:
            ordered_dims = ordered_dims_match
            # ... (Logic to sort dims based on aggregator weights)
            try:
                w = aggregator.weight.detach().cpu().numpy().reshape(-1)
                ranks = sorted(list(enumerate(w.tolist())), key=lambda t: t[1], reverse=True)
                index_to_dim = {}
                dim_to_index = {}
                for pos, (idx, _) in enumerate(ranks):
                    if pos < len(ordered_dims):
                        dim_name = ordered_dims[pos]
                        index_to_dim[idx] = dim_name
                        dim_to_index[dim_name] = idx
            except Exception:
                index_to_dim = {i: ordered_dims_match[i] for i in range(len(ordered_dims_match))}
                dim_to_index = {dim: i for i, dim in enumerate(ordered_dims_match)}

        # Move to device
        param = next(model.parameters())
        device = param.device
        dtype = param.dtype
        # Access the head through the model wrapper
        if hasattr(model.base_model, "score"):
            model.base_model.score.to(device=device, dtype=dtype)
        elif hasattr(model.base_model, "classifier"):
            model.base_model.classifier.to(device=device, dtype=dtype)
            
        aggregator.to(device=device, dtype=dtype)
        aggregator.eval()
        
        # Log aggregator weights
        try:
            w = aggregator.weight.detach().cpu().numpy().reshape(-1)
            print(f"Aggregator weight sum={float(w.sum()):.6f}, dim={w.shape[0]}")
            print(f"Aggregator weights={w.tolist()}")
        except Exception:
            pass

    # Final eval mode
    model.eval()

    # Ensure scalar custom head is in correct dtype/device
    if is_scalar_custom:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
        if hasattr(model.base_model, "score"):
            model.base_model.score.to(device=device, dtype=dtype)
        elif hasattr(model.base_model, "classifier"):
            model.base_model.classifier.to(device=device, dtype=dtype)
    
    return model, tokenizer, aggregator, index_to_dim, ordered_dims, dim_to_index


def setup_devices(model, device_ids_str: str):
    device_ids = [int(x) for x in device_ids_str.split(',') if x.strip() != '']
    if torch.cuda.is_available() and len(device_ids) > 0:
        primary = device_ids[0]
        model.to(f"cuda:{primary}")
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.eval()
        return model
    else:
        model.eval()
        return model


def load_template(template_path):
    template_dir = os.path.dirname(template_path) or "."
    template_file = os.path.basename(template_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters["tojson"] = lambda obj: json.dumps(obj)
    return env.get_template(template_file)


def evaluate_prompt(model, tokenizer, aggregator, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    if not hasattr(outputs, "logits") or outputs.logits is None:
        raise RuntimeError("模型输出不包含 logits；请确保使用的是序列分类模型并已接入自定义头。")
    logits = outputs.logits
    
    # Handle scalar mode (aggregator is None)
    if aggregator is None:
        score = logits.squeeze(-1)
    else:
        score = aggregator(logits).squeeze(-1)
        
    reward = score.detach().cpu().item()
    return reward


HEADER = (
    "你是一个优秀的语言表达者，请基于下面的情境给出发言。\n\n"
    "情境：\n"
    "现在的情境是一个辩论赛，给定辩题与持方，同时提供双方的一辩稿、以及对方最新一轮发言。\n"
    "（A）发现对方论点与证据中的漏洞/盲点，以及对方对己方立场的攻击，确定核心分歧；\n"
    "（B）整合与引用可靠资料以支持己方论点或驳斥对方；\n"
    "（C）组织并书写一段连贯的陈词/回应（包含拆解对方漏洞与回扣、阐释己方主张），并给出内容完整、行文流畅的陈词进行回应。\n\n"
    "输入占位（你将收到）：\n"
    "<|背景信息开始|>\n"
)


def build_input_from_match(item: dict) -> str:
    topic = item.get("辩题", "")
    stance = item.get("持方", "")
    pro_first = item.get("正方一辩稿", "")
    con_first = item.get("反方一辩稿", "")
    opp_latest = item.get("对方发言", "")
    parts = [
        HEADER,
        f"辩题：{topic}\n",
        f"持方：{stance}\n",
        f"正方一辩稿：{pro_first}：反方一辩稿：{con_first}\n",
        "<|背景信息结束|>\n\n",
        "<|上一轮发言开始|>\n",
        f"对方发言：{opp_latest}\n",
        "<|上一轮发言结束|>\n\n",
        "请在以上背景与上一轮发言的基础上生成本轮陈词，在1300字以内论述相关内容\n",
    ]
    return "".join(parts)


def flatten_match_structure(match_data):
    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    for it in v:
                        yield it
                else:
                    yield from walk(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from walk(it)
    yield from walk(match_data)


def convert_match_to_rm_examples(match_data):
    examples = []
    for item in flatten_match_structure(match_data):
        inp = build_input_from_match(item)
        outp = item.get("本轮发言", "")
        rec = {"input": inp, "output": outp}
        if "value" in item:
            try:
                rec["value"] = float(item.get("value"))
            except Exception:
                rec["value"] = item.get("value")
        if "values" in item:
            rec["values"] = item.get("values")
        examples.append(rec)
    return examples


def ensure_rm_format(example_data):
    if isinstance(example_data, list) and len(example_data) > 0:
        sample = example_data[0]
        if isinstance(sample, dict) and ("input" in sample and "output" in sample):
            return example_data
    return convert_match_to_rm_examples(example_data)


def score_items(model, tokenizer, template, items, aggregator, index_to_dim, ordered_dims=None, dim_to_index=None, batch_size=16, store_ground_truth=False, save_gt_values=False):
    results = []
    device = next(model.parameters()).device
    batch_texts = []
    batch_indices = []
    for idx, example in enumerate(items):
        rendered_prompt = template.render(
            messages=[
                {"role": "user", "content": example.get("input", "")},
                {"role": "assistant", "content": example.get("output", "")},
            ],
            add_generation_prompt=False,
        ).strip()
        batch_texts.append(rendered_prompt)
        batch_indices.append(idx)
        if len(batch_texts) == batch_size or idx == len(items) - 1:
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            
            if aggregator is None:
                # Scalar mode
                scores = logits.squeeze(-1).to(torch.float32).detach().cpu().tolist()
                vecs = scores  # In scalar mode, vector is the score itself
            else:
                # Multi-dim mode
                logits = logits.to(aggregator.weight.dtype)
                scores = aggregator(logits).squeeze(-1).to(torch.float32).detach().cpu().tolist()
                vecs = logits.to(torch.float32).detach().cpu().tolist()
                
            if not isinstance(scores, list):
                scores = [scores]
            if not isinstance(vecs, list):
                vecs = [vecs]
                
            for bi, s in zip(batch_indices, scores):
                record = {
                    "input": items[bi].get("input", ""),
                    "output": items[bi].get("output", ""),
                    "value": float(s),
                }
                try:
                    vec = vecs[batch_indices.index(bi)]
                    if aggregator is None:
                        # For scalar mode, we can just store the scalar score if needed, or skip values
                        record["values"] = float(vec)
                    else:
                        if isinstance(vec, list):
                            if index_to_dim and ordered_dims and dim_to_index:
                                record["values"] = {dim: float(vec[dim_to_index[dim]]) for dim in ordered_dims}
                            elif index_to_dim:
                                record["values"] = {index_to_dim.get(j, str(j)): float(vec[j]) for j in range(len(vec))}
                            else:
                                record["values"] = [float(v) for v in vec]
                        else:
                            record["values"] = vec
                except Exception:
                    record["values"] = vecs[batch_indices.index(bi)]
                if store_ground_truth:
                    if save_gt_values and ("values" in items[bi]):
                        gt_vals = items[bi]["values"]
                        if isinstance(gt_vals, dict) and len(gt_vals) == 8:
                            record["ground_truth"] = gt_vals
                        elif isinstance(gt_vals, list) and len(gt_vals) == 8:
                            record["ground_truth"] = [float(v) for v in gt_vals]
                        elif "value" in items[bi]:
                            gt = items[bi]["value"]
                            try:
                                gt = float(gt)
                            except Exception:
                                pass
                            record["ground_truth"] = gt
                    elif "value" in items[bi]:
                        gt = items[bi]["value"]
                        try:
                            gt = float(gt)
                        except Exception:
                            pass
                        record["ground_truth"] = gt
                results.append(record)
            batch_texts = []
            batch_indices = []
    return results


def main():
    args = parse_args()
    model, tokenizer, aggregator, index_to_dim, ordered_dims, dim_to_index = load_model_and_tokenizer(args)
    model = setup_devices(model, args.device_ids)
    param = next(model.parameters())
    if aggregator is not None:
        aggregator.to(device=param.device, dtype=param.dtype)
        aggregator.eval()
    with open(args.example_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    example_data = ensure_rm_format(raw_data)
    if not isinstance(example_data, list):
        raise ValueError("输入 JSON 必须是列表。")
    template = load_template(args.template_path)
    gt_values_paths = {
        "/Users/woyu/Desktop/learning_material/25_a/SII/edu/code/sotopia-rl/data/creativity/real/merged_all.json",
        "/Users/woyu/Desktop/learning_material/25_a/SII/edu/code/sotopia-rl/data/creativity/real/merged_to_creativity_real.json",
    }
    save_gt_values = os.path.abspath(args.example_path) in gt_values_paths
    results = score_items(
        model,
        tokenizer,
        template,
        example_data,
        aggregator,
        index_to_dim,
        ordered_dims,
        dim_to_index,
        batch_size=args.batch_size,
        store_ground_truth=args.store_ground_truth,
        save_gt_values=save_gt_values,
    )
    out_dir = os.path.dirname(args.output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} scored items to: {args.output_path}")


if __name__ == "__main__":
    main()
