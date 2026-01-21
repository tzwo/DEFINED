import os
import json

import torch
import torch._dynamo
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
from jinja2 import Environment, FileSystemLoader
from peft import LoraConfig, PeftModelForSequenceClassification
from torch.nn import MSELoss
from torch.utils.data import random_split, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from torch.utils.data import Dataset

torch._dynamo.config.suppress_errors = False

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_MODE"] = "offline"
os.environ["NCCL_SHM_DISABLE"] = "1"

class DEFINEDClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.scoring_layer = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(2048, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(),
            nn.Linear(1024, 1024, bias=False), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(1024, output_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scoring_layer(x)


# NOTE: We switch to AutoModelForSequenceClassification + PEFT wrapper.
# The classification head is replaced by DEFINEDClassifier following train_policy.py.


class DebateRMTrainer(Trainer):
    def __init__(self, args, accelerator, **kwargs):
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        self.seed = getattr(args, "seed", 42)
        set_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.padding_side = "right"
        
        self.num_labels = getattr(args, "num_labels", 8)
        self.scalar_mode = bool(getattr(args, "scalar_rm", False))
        self.mixed_data_switch = False
        self.oversample_k = getattr(args, "oversample_k", 1)
        
        train_dataset, eval_dataset = self.setup_dataset(tokenizer)

        if self.accelerator.is_main_process:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                mode="offline",
                config={k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))},
            )

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(","),
        )
        # Initialize model according to mode
        if self.scalar_mode:
            base_cls = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=1,
                torch_dtype="auto",
                pad_token_id=tokenizer.pad_token_id,
            )
            base_cls.config.pad_token_id = tokenizer.pad_token_id
            self.model = PeftModelForSequenceClassification(base_cls, peft_config)
            self.base_model = None
            self.aggregator = None
            if self.accelerator.is_main_process:
                print("[Mode] Scalar RM mode enabled: training single-value predictor (no custom head)")
        else:
            base_cls = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                torch_dtype="auto",
                num_labels=self.num_labels,
                pad_token_id=tokenizer.pad_token_id,
            )
            base_cls.config.pad_token_id = tokenizer.pad_token_id
            adapter_path = getattr(self.args, "adapter_path", None)
            if adapter_path:
                adapter_stub = os.path.join(adapter_path, "adapter_model")
                has_adapter = os.path.exists(adapter_stub + ".safetensors") or os.path.exists(adapter_stub + ".bin")
                if has_adapter:
                    self.base_model = PeftModelForSequenceClassification.from_pretrained(base_cls, adapter_path, is_trainable=True)
                else:
                    pt_bin = os.path.join(adapter_path, "pytorch_model.bin")
                    if os.path.exists(pt_bin):
                        state = torch.load(pt_bin, map_location="cpu")
                        base_cls.load_state_dict(state, strict=False)
                    self.base_model = PeftModelForSequenceClassification(base_cls, peft_config)
                rm_head_path = os.path.join(adapter_path, "rm_head.pt")
                if os.path.exists(rm_head_path):
                    head_state = torch.load(rm_head_path, map_location="cpu")
                    self.num_labels = int(head_state.get("num_labels", self.num_labels))
                    hidden_size = getattr(self.base_model.base_model.config, "hidden_size", None) or getattr(self.base_model.base_model.config, "hidden_sizes", [])[0]
                    self.base_model.base_model.score = DEFINEDClassifier(hidden_size, output_dim=self.num_labels)
                    self.base_model.base_model.score.load_state_dict(head_state["head"], strict=True)
                else:
                    hidden_size = getattr(self.base_model.base_model.config, "hidden_size", None) or getattr(self.base_model.base_model.config, "hidden_sizes", [])[0]
                    self.base_model.base_model.score = DEFINEDClassifier(hidden_size, output_dim=self.num_labels)
            else:
                self.base_model = PeftModelForSequenceClassification(base_cls, peft_config)
                hidden_size = getattr(self.base_model.base_model.config, "hidden_size", None) or getattr(self.base_model.base_model.config, "hidden_sizes", [])[0]
                self.base_model.base_model.score = DEFINEDClassifier(hidden_size, output_dim=self.num_labels)
            self.aggregator = nn.Linear(self.num_labels, 1, bias=False)
            for p in self.base_model.base_model.score.parameters():
                p.requires_grad = True
            for p in self.aggregator.parameters():
                p.requires_grad = True
            self.model = AggregatedSequenceClassification(self.base_model, self.aggregator)

        training_args = TrainingArguments(
            output_dir=args.checkpoint_dir,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.val_batch_size,
            num_train_epochs=args.num_epochs,
            logging_steps=1,
            save_steps=args.evaluation_steps,
            save_strategy="steps",
            logging_dir="./logs",
            gradient_accumulation_steps=args.accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=int(len(train_dataset) * args.warmup_epochs),
            optim="adamw_torch",
            remove_unused_columns=False,
            dataloader_num_workers=4,
            report_to="wandb",
            ddp_find_unused_parameters=True,
            eval_strategy="steps",
            eval_steps=args.evaluation_steps,
            label_names=["labels"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.seed,
        )

        if hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "collate_fn"):
            collate_fn = train_dataset.dataset.collate_fn
        elif hasattr(train_dataset, "collate_fn"):
            collate_fn = train_dataset.collate_fn
        else:
            collate_fn = None

        super().__init__(
            model=self.model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
            compute_metrics=self.compute_metrics,
            **kwargs,
        )
        self.loss_fn = MSELoss()

        # Regularization settings
        # 2) Constrain the per-dimension values multiplied by weights (logits6) into [value_lower, value_upper]
        self.value_lower = getattr(args, "value_lower", 0.0)
        self.value_upper = getattr(args, "value_upper", 100.0)
        self.value_range_reg = getattr(args, "value_range_reg", 0)
        # 3) Align each of the 6 logits to the scalar label (per-dim MSE)
        self.dim_align_reg = getattr(args, "dim_align_reg", 0)
        self.agg_range_reg = getattr(args, "agg_range_reg", 1.0)
        self.agg_lower = 0.0
        self.agg_upper = 100.0
        self.oversample_k = getattr(args, "oversample_k", 1)

        # Ensure aggregator device/dtype matches model
        if not self.scalar_mode:
            params = list(self.base_model.parameters())
            if len(params) > 0:
                self.aggregator.to(device=self.device, dtype=params[0].dtype)
            else:
                self.aggregator.to(device=self.device)

    def setup_dataset(self, tokenizer):
        env = Environment(loader=FileSystemLoader("/".join(self.args.template_path.split("/")[:-1])))
        template = env.get_template(self.args.template_path.split("/")[-1])
        val_path = getattr(self.args, "val_path", None)
        if val_path is None:
            match_path = self.args.reward_data_path
            dataset = MatchRMDataset(match_path, tokenizer, template, self.args.max_length, self.num_labels, oversample_k=self.oversample_k)
            if getattr(dataset, "has_mixed", False):
                self.mixed_data_switch = True
                if self.accelerator.is_main_process:
                    print("Mixed data detected! Enabling mixed training switch.")
            if self.accelerator.is_main_process:
                print(f"dataset: {len(dataset)}")
            all_match_keys = []
            for it in dataset.data:
                mk = it.get("match_key", None)
                if mk is not None:
                    all_match_keys.append(mk)
            uniq = sorted(list(set(all_match_keys)))
            rnd = random.Random(self.seed)
            rnd.shuffle(uniq)
            val_match_count = max(1, int(len(uniq) * 0.05))
            val_match_set = set(uniq[:val_match_count])
            train_indices = [i for i, it in enumerate(dataset.data) if it.get("match_key") not in val_match_set]
            val_indices = [i for i, it in enumerate(dataset.data) if it.get("match_key") in val_match_set]
            if self.accelerator.is_main_process:
                print(f"matches_total={len(uniq)} matches_val={len(val_match_set)} train_items={len(train_indices)} val_items={len(val_indices)}")
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            return train_dataset, val_dataset
        else:
            train_dataset = MatchRMDataset(self.args.reward_data_path, tokenizer, template, self.args.max_length, self.num_labels, oversample_k=self.oversample_k)
            if getattr(train_dataset, "has_mixed", False):
                self.mixed_data_switch = True
                if self.accelerator.is_main_process:
                    print("Mixed data detected in train set! Enabling mixed training switch.")
            with open(val_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            has_values = False
            if isinstance(raw, list):
                for obj in raw:
                    if isinstance(obj, dict) and ("values" in obj) and isinstance(obj["values"], (dict, list)):
                        has_values = True
                        break
                if not has_values:
                    for obj in raw:
                        if isinstance(obj, dict):
                            for mk, rounds in obj.items():
                                if isinstance(rounds, list):
                                    for rec in rounds:
                                        if isinstance(rec, dict) and ("values" in rec) and isinstance(rec["values"], (dict, list)):
                                            has_values = True
                                            break
                                    if has_values:
                                        break
                        if has_values:
                            break
            elif isinstance(raw, dict):
                for mk, rounds in raw.items():
                    if isinstance(rounds, list):
                        for rec in rounds:
                            if isinstance(rec, dict) and ("values" in rec) and isinstance(rec["values"], (dict, list)):
                                has_values = True
                                break
                        if has_values:
                            break
                    elif isinstance(rounds, dict) and ("values" in rounds) and isinstance(rounds["values"], (dict, list)):
                        has_values = True
                        break
            if has_values:
                val_dataset = MultiValueRMDataset(val_path, tokenizer, template, self.args.max_length, self.num_labels)
            else:
                val_dataset = MatchRMDataset(val_path, tokenizer, template, self.args.max_length, self.num_labels)
            return train_dataset, val_dataset

    def train(self, *args, **kwargs):
        if self.accelerator.is_main_process:
            if self.scalar_mode:
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"Training started (scalar mode): trainable params={total_params}")
            else:
                head_params = sum(p.numel() for p in self.base_model.base_model.score.parameters() if p.requires_grad)
                agg_params = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
                print(f"Training started: custom head trainable params={head_params}, aggregator trainable params={agg_params}")
                self.base_model.print_trainable_parameters()
        return super().train(*args, **kwargs)

    def create_optimizer(self):
        super().create_optimizer()
        if self.scalar_mode:
            return
        opt = getattr(self, "optimizer", None)
        agg_params = [p for p in self.aggregator.parameters() if p.requires_grad]
        agg_ids = {id(p) for p in agg_params}
        if opt is not None:
            base_lr = opt.param_groups[0].get("lr", None)
            if base_lr is None:
                base_lr = getattr(self.args, "learning_rate", 5e-5)
            base_wd = opt.param_groups[0].get("weight_decay", 0.0)
            for g in opt.param_groups:
                kept = []
                for p in g["params"]:
                    if id(p) not in agg_ids:
                        kept.append(p)
                g["params"] = kept
            opt.add_param_group({"params": agg_params, "lr": base_lr * 50, "weight_decay": base_wd})
            print(f"Aggregator params assigned to separate group with lr={base_lr * 50} weight_decay={base_wd}")
        else:
            print("Optimizer is None after creation")

    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     loss = super().training_step(model, inputs, num_items_in_batch)
    #     try:
    #         for name, p in self.aggregator.named_parameters():
    #             gn = None if p.grad is None else float(p.grad.data.norm().item())
    #             # print(f"Aggregator grad {name} norm={gn}")
    #     except Exception as e:
    #         print(f"Aggregator grad check error: {e}")
    #     return loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_masks = inputs["attention_mask"]
        true_rewards = inputs["labels"]
        is_multidim = inputs.get("is_multidim", None)

        if self.scalar_mode:
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            predicted = outputs.logits.squeeze(-1).to(torch.float32)
            true_rewards = true_rewards.to(torch.float32)
            if true_rewards.dim() == 2:
                if true_rewards.size(-1) == 1:
                    true_rewards = true_rewards.squeeze(-1)
                else:
                    true_rewards = true_rewards.mean(dim=-1)
            loss = self.loss_fn(predicted, true_rewards)
            return (loss, outputs) if return_outputs else loss
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_masks, use_cache=False)
            logits = outputs["logits"]
            score = outputs["score"]
            weight_sum = outputs["weight_sum"]
            score = score.to(torch.float32)
            true_rewards = true_rewards.to(torch.float32)
            
            if (not self.model.training) and true_rewards.dim() == 2 and true_rewards.size(-1) == self.num_labels:
                dims_mse = F.mse_loss(logits.to(torch.float32), true_rewards.to(torch.float32), reduction="mean")
                total_loss = dims_mse
                return (total_loss, score) if return_outputs else total_loss
            
            scalar_target = true_rewards
            if true_rewards.dim() == 2:
                if true_rewards.size(-1) == self.num_labels:
                    scalar_target = true_rewards.mean(dim=-1)
                elif true_rewards.size(-1) == 1:
                    scalar_target = true_rewards.squeeze(-1)
                else:
                    raise ValueError(f"Unsupported labels shape: {true_rewards.shape}")
            elif true_rewards.dim() == 1:
                pass
            else:
                raise ValueError(f"Unsupported labels shape: {true_rewards.shape}")

            if self.mixed_data_switch and is_multidim is not None:
                loss_scalar = (score - scalar_target) ** 2
                loss_multidim = ((logits.to(torch.float32) - inputs["labels"].to(torch.float32)) ** 2).sum(dim=-1)
                combined_loss = (1.0 - is_multidim) * loss_scalar + is_multidim * loss_multidim
                base_mse = combined_loss.mean()
            else:
                base_mse = self.loss_fn(score, scalar_target)

            lower_viol_vals = torch.nn.functional.relu(self.value_lower - logits)
            upper_viol_vals = torch.nn.functional.relu(logits - self.value_upper)
            value_range_penalty = (lower_viol_vals.pow(2) + upper_viol_vals.pow(2)).mean()
            loss_value_range = (self.value_range_reg * value_range_penalty.to(base_mse.dtype)) if (self.value_range_reg and self.value_range_reg > 0) else torch.zeros((), dtype=base_mse.dtype, device=base_mse.device)
            agg_lower_viol = torch.nn.functional.relu(self.agg_lower - logits)
            agg_upper_viol = torch.nn.functional.relu(logits - self.agg_upper)
            agg_range_penalty = (agg_lower_viol.pow(2) + agg_upper_viol.pow(2)).mean()
            loss_agg_range = (self.agg_range_reg * agg_range_penalty.to(base_mse.dtype))
            targetn = scalar_target.unsqueeze(-1).expand_as(logits)
            dim_align_mse = F.mse_loss(logits.to(torch.float32), targetn.to(torch.float32), reduction="mean")
            loss_dim_align = (self.dim_align_reg * dim_align_mse.to(base_mse.dtype)) if (self.dim_align_reg and self.dim_align_reg > 0) else torch.zeros((), dtype=base_mse.dtype, device=base_mse.device)
            total_loss = base_mse + loss_value_range + loss_dim_align + loss_agg_range
            gs = getattr(self.state, "global_step", None)
            if gs is not None and (gs % max(1, self.args.logging_steps) == 0):
                metrics = {
                    "loss_total": total_loss.detach().item(),
                    "loss_mse": base_mse.detach().item(),
                    "loss_value_range": loss_value_range.detach().item(),
                    "loss_agg_range": loss_agg_range.detach().item(),
                    "loss_dim_align": loss_dim_align.detach().item(),
                    "agg_weight_sum": weight_sum.detach().item(),
                }
                self.log(metrics)
                if self.accelerator.is_main_process:
                    print(
                        f"[step {gs}] total={metrics['loss_total']:.6f} "
                        f"mse={metrics['loss_mse']:.6f} wsum={metrics['agg_weight_sum']:.6f} "
                        f"loss_vrange={metrics['loss_value_range']:.6f} "
                        f"loss_agg={metrics['loss_agg_range']:.6f} "
                        f"loss_align={metrics['loss_dim_align']:.6f}"
                    )
            return (total_loss, score) if return_outputs else total_loss

    def compute_metrics(self, eval_preds):
        from transformers.trainer_utils import EvalPrediction
        if EvalPrediction is not None and isinstance(eval_preds, EvalPrediction):
            preds = eval_preds.predictions
            labels = eval_preds.label_ids
        else:
            preds, labels = eval_preds
        preds = preds.astype(np.float32)
        labels = labels.astype(np.float32)
        if self.scalar_mode:
            pred_score = preds.squeeze()
            labels_scalar = labels.squeeze()
        else:
            w = self.aggregator.weight.detach().float().cpu()
            pos_w = torch.nn.functional.softplus(w).float().numpy()
            s = float(pos_w.sum())
            if s > 0:
                pos_w = pos_w / s
            else:
                pos_w = np.ones_like(pos_w, dtype=np.float32) / max(1, pos_w.size)
            if preds.ndim == 2 and preds.shape[-1] == self.num_labels:
                pred_score = preds @ pos_w
            else:
                pred_score = preds.squeeze()
            if labels.ndim == 2:
                if labels.shape[-1] == self.num_labels:
                    labels_scalar = labels.mean(axis=-1)
                elif labels.shape[-1] == 1:
                    labels_scalar = labels.squeeze(-1)
                else:
                    labels_scalar = labels.squeeze()
            else:
                labels_scalar = labels.squeeze()
        pred_score = np.asarray(pred_score, dtype=np.float32).reshape(-1)
        labels_scalar = np.asarray(labels_scalar, dtype=np.float32).reshape(-1)
        min_len = min(pred_score.shape[0], labels_scalar.shape[0])
        if pred_score.shape[0] != labels_scalar.shape[0]:
            pred_score = pred_score[:min_len]
            labels_scalar = labels_scalar[:min_len]
        mse = float(np.mean((pred_score - labels_scalar) ** 2))
        def round_to_tens(x):
            return np.round(x / 10.0) * 10.0
        tens_consistency = float(np.mean(round_to_tens(pred_score) == round_to_tens(labels_scalar)))
        return {"mse_loss": mse, "round_consistency": tens_consistency}

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        """
        Save model checkpoint and ensure PEFT adapter and custom head parameters
        are saved to the same directory for inference compatibility.
        """
        # Resolve output directory
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # First, let Trainer perform its default save (e.g., training args, state)
        super().save_model(output_dir, _internal_call=_internal_call)
        if self.scalar_mode:
            self.model.save_pretrained(output_dir)
            if self.accelerator.is_main_process:
                print(f"Saved scalar PEFT model to: {output_dir}")
        else:
            self.base_model.save_pretrained(output_dir)
            if self.accelerator.is_main_process:
                print(f"Saved PEFT adapter to: {output_dir}")
            agg_state = {}
            if hasattr(self.aggregator, "weight") and self.aggregator.weight is not None:
                w = self.aggregator.weight.detach().cpu()
                pos_w = torch.nn.functional.softplus(w)
                s = pos_w.sum()
                agg_state["weight"] = pos_w / (s + 1e-12)
            head_state = {
                "head": self.base_model.base_model.score.state_dict(),
                "aggregator": agg_state,
                "num_labels": self.num_labels,
            }
            torch.save(head_state, os.path.join(output_dir, "rm_head.pt"))
            if self.accelerator.is_main_process:
                print(f"Saved RM head params to: {os.path.join(output_dir, 'rm_head.pt')}")
class AggregatedSequenceClassification(nn.Module):
    def __init__(self, base_model: nn.Module, aggregator: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.aggregator = aggregator
    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, **kwargs)
        logits = outputs.logits
        if logits.dtype != self.aggregator.weight.dtype:
            logits = logits.to(self.aggregator.weight.dtype)
        pos_weight = torch.nn.functional.softplus(self.aggregator.weight)
        weight_sum = pos_weight.sum()
        norm_weight = pos_weight / (weight_sum + 1e-12)
        score = torch.nn.functional.linear(logits, norm_weight).squeeze(-1)
        return {"logits": logits, "score": score, "weight_sum": weight_sum, "norm_weight": norm_weight}

class MatchRMDataset(Dataset):
    def __init__(self, match_data_path: str, tokenizer, template, max_length: int, num_labels: int = 8, oversample_k: int = 1):
        self.num_labels = num_labels
        self.oversample_k = oversample_k
        self.data, self.has_mixed, self.has_multidim, self.has_scalar = self._load_and_convert(match_data_path, num_labels, oversample_k)
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length

    @staticmethod
    def _load_and_convert(file_path: str, num_labels: int, oversample_k: int = 1):
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = []
        has_multidim = False
        has_scalar = False

        def build_input(topic: str, side: str, pos1: str, neg1: str, opp: str) -> str:
            return (
                "你是一个优秀的语言表达者，请基于下面的情境给出发言。\n\n"
                "情境：\n"
                "现在的情境是一个辩论赛，给定辩题与持方，同时提供双方的一辩稿、以及对方最新一轮发言。\n"
                "（A）发现对方论点与证据中的漏洞/盲点，以及对方对己方立场的攻击，确定核心分歧；\n"
                "（B）整合与引用可靠资料以支持己方论点或驳斥对方；\n"
                "（C）组织并书写一段连贯的陈词/回应（包含拆解对方漏洞与回扣、阐释己方主张），并给出内容完整、行文流畅的陈词进行回应。\n\n"
                "输入占位（你将收到）：\n"
                "<|背景信息开始|>\n"
                f"辩题：{topic}\n"
                f"持方：{side}\n"
                f"正方一辩稿：{pos1}\n"
                f"反方一辩稿：{neg1}\n"
                "<|背景信息结束|>\n\n"
                "<|上一轮发言开始|>\n"
                f"对方发言：{opp}\n"
                "<|上一轮发言结束|>\n\n"
                "请在以上背景与上一轮发言的基础上生成本轮陈词，在1300字以内论述相关内容\n"
            )

        def convert_record(rec: dict, match_key: str):
            topic = rec.get("辩题", "")
            side = rec.get("持方", "")
            pos1 = rec.get("正方一辩稿", rec.get("正方一辩", ""))
            neg1 = rec.get("反方一辩稿", rec.get("反方一辩", ""))
            opp = rec.get("对方发言", rec.get("上一轮发言", ""))
            out = rec.get("本轮发言", rec.get("发言", ""))
            
            is_multidim = 0.0
            labels = [0.0] * num_labels
            
            if "values" in rec and isinstance(rec["values"], (dict, list)):
                is_multidim = 1.0
                v = rec["values"]
                if isinstance(v, dict):
                    vec = [float(v.get(k, 0.0)) for k in list(v.keys())][:num_labels]
                elif isinstance(v, list):
                    vec = [float(x) for x in v][:num_labels]
                else:
                    vec = [0.0] * num_labels
                if len(vec) < num_labels:
                    vec = vec + [0.0] * (num_labels - len(vec))
                labels = vec
            else:
                val = rec.get("value", rec.get("score", None))
                value = float(val) if val is not None else 0.0
                labels = [value] * num_labels
            
            return {
                "input": build_input(str(topic), str(side), str(pos1), str(neg1), str(opp)),
                "output": str(out),
                "labels": labels,
                "is_multidim": is_multidim,
                "match_key": str(match_key),
            }

        if isinstance(raw, list):
            for obj in raw:
                if isinstance(obj, dict):
                    for mk, rounds in obj.items():
                        if isinstance(rounds, list):
                            for rec in rounds:
                                if isinstance(rec, dict):
                                    it = convert_record(rec, mk)
                                    if it["is_multidim"] > 0.5:
                                        has_multidim = True
                                        if oversample_k > 1:
                                            for _ in range(oversample_k):
                                                items.append(it)
                                        else:
                                            items.append(it)
                                    else:
                                        has_scalar = True
                                        items.append(it)
        elif isinstance(raw, dict):
            for mk, rounds in raw.items():
                if isinstance(rounds, list):
                    for rec in rounds:
                        if isinstance(rec, dict):
                            it = convert_record(rec, mk)
                            if it["is_multidim"] > 0.5:
                                has_multidim = True
                                if oversample_k > 1:
                                    for _ in range(oversample_k):
                                        items.append(it)
                                else:
                                    items.append(it)
                            else:
                                has_scalar = True
                                items.append(it)

        return items, (has_multidim and has_scalar), has_multidim, has_scalar

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        rendered_text = self.template.render(
            messages=[
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]},
            ],
            add_generation_prompt=False,
        ).strip()

        tokenized_input = self.tokenizer(
            rendered_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        assert tokenized_input["input_ids"][0][-1] == self.tokenizer.eos_token_id

        return {
            "input_ids": tokenized_input["input_ids"].squeeze(),
            "attention_mask": tokenized_input["attention_mask"].squeeze(),
            "labels": torch.tensor(item["labels"], dtype=torch.float32),
            "is_multidim": torch.tensor(item["is_multidim"], dtype=torch.float32)
        }

    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0,
        )
        labels = torch.stack([b["labels"] for b in batch])
        is_multidim = torch.stack([b["is_multidim"] for b in batch])
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "is_multidim": is_multidim}

class MultiValueRMDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, template, max_length: int, num_labels: int):
        self.data = self._load_and_convert(data_path, num_labels)
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.num_labels = num_labels
    @staticmethod
    def _load_and_convert(file_path: str, num_labels: int):
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = []
        def build_input(topic: str, side: str, pos1: str, neg1: str, opp: str) -> str:
            return (
                "你是一个优秀的语言表达者，请基于下面的情境给出发言。\n\n"
                "情境：\n"
                "现在的情境是一个辩论赛，给定辩题与持方，同时提供双方的一辩稿、以及对方最新一轮发言。\n"
                "（A）发现对方论点与证据中的漏洞/盲点，以及对方对己方立场的攻击，确定核心分歧；\n"
                "（B）整合与引用可靠资料以支持己方论点或驳斥对方；\n"
                "（C）组织并书写一段连贯的陈词/回应（包含拆解对方漏洞与回扣、阐释己方主张），并给出内容完整、行文流畅的陈词进行回应。\n\n"
                "输入占位（你将收到）：\n"
                "<|背景信息开始|>\n"
                f"辩题：{topic}\n"
                f"持方：{side}\n"
                f"正方一辩稿：{pos1}\n"
                f"反方一辩稿：{neg1}\n"
                "<|背景信息结束|>\n\n"
                "<|上一轮发言开始|>\n"
                f"对方发言：{opp}\n"
                "<|上一轮发言结束|>\n\n"
                "请在以上背景与上一轮发言的基础上生成本轮陈词，在1300字以内论述相关内容\n"
            )
        def convert_record_nested(rec: dict, match_key: str):
            topic = rec.get("辩题", "")
            side = rec.get("持方", "")
            pos1 = rec.get("正方一辩稿", rec.get("正方一辩", ""))
            neg1 = rec.get("反方一辩稿", rec.get("反方一辩", ""))
            opp = rec.get("对方发言", rec.get("上一轮发言", ""))
            out = rec.get("本轮发言", rec.get("发言", ""))
            vdict = rec.get("values", {})
            vec = [float(vdict.get(k, 0.0)) for k in list(vdict.keys())][:num_labels]
            if len(vec) < num_labels:
                vec = vec + [0.0] * (num_labels - len(vec))
            return {
                "input": build_input(str(topic), str(side), str(pos1), str(neg1), str(opp)),
                "output": str(out),
                "values": vec,
                "match_key": str(match_key),
            }
        def convert_record_flat(obj: dict):
            inp = obj.get("input", "")
            out = obj.get("output", "")
            v = obj.get("values", {})
            if isinstance(v, dict):
                vec = [float(v.get(k, 0.0)) for k in list(v.keys())][:num_labels]
            elif isinstance(v, list):
                vec = [float(x) for x in v][:num_labels]
            else:
                vec = [0.0] * num_labels
            if len(vec) < num_labels:
                vec = vec + [0.0] * (num_labels - len(vec))
            return {
                "input": str(inp),
                "output": str(out),
                "values": vec,
                "match_key": str(obj.get("match_key", "")),
            }
        if isinstance(raw, list):
            # Support flat list of records with 'input'/'output'/'values'
            if len(raw) > 0 and isinstance(raw[0], dict) and ("input" in raw[0]) and ("values" in raw[0]):
                for obj in raw:
                    if isinstance(obj, dict):
                        items.append(convert_record_flat(obj))
            else:
                for obj in raw:
                    if isinstance(obj, dict):
                        for mk, rounds in obj.items():
                            if isinstance(rounds, list):
                                for rec in rounds:
                                    if isinstance(rec, dict):
                                        items.append(convert_record_nested(rec, mk))
        elif isinstance(raw, dict):
            # Either nested dict of match_key -> list[rec], or a single flat record
            if ("input" in raw) and ("values" in raw):
                items.append(convert_record_flat(raw))
            else:
                for mk, rounds in raw.items():
                    if isinstance(rounds, list):
                        for rec in rounds:
                            if isinstance(rec, dict):
                                items.append(convert_record_nested(rec, mk))
        return items
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int):
        item = self.data[idx]
        rendered_text = self.template.render(
            messages=[
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]},
            ],
            add_generation_prompt=False,
        ).strip()
        tokenized_input = self.tokenizer(
            rendered_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        assert tokenized_input["input_ids"][0][-1] == self.tokenizer.eos_token_id
        return {
            "input_ids": tokenized_input["input_ids"].squeeze(),
            "attention_mask": tokenized_input["attention_mask"].squeeze(),
            "labels": torch.tensor(item["values"], dtype=torch.float32),
            "is_multidim": torch.tensor(1.0, dtype=torch.float32),
        }
    def collate_fn(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0,
        )
        labels = torch.stack([b["labels"] for b in batch])
        is_multidim = torch.stack([b["is_multidim"] for b in batch])
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "is_multidim": is_multidim}
