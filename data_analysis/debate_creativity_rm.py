import argparse

from accelerate import Accelerator

from debate_rm_creativity_trainer import DebateRMTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a debate reward model with a 6-dim head linearly aggregated to a single score, using LoRA.")

    # Model and optimizer settings
    parser.add_argument("--model_name", type=str, default="/data/models/gemma-2-2b-it", help="Path or name of the base model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")

    # Data and template paths
    parser.add_argument("--reward_data_path", type=str, required=True, help="Path to the reward data JSON file")
    parser.add_argument("--template_path", type=str, required=True, help="Path to the Jinja template file")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to load a LoRA adapter and RM head")
    parser.add_argument("--val_path", type=str, default=None, help="Optional validation data JSON path")
    parser.add_argument("--oversample_k", type=int, default=1, help="Oversample factor for multidimensional data")

    # Training schedule
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training per device")
    parser.add_argument("--val_batch_size", type=int, default=1, help="Batch size for evaluation per device")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum length for tokenized inputs")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--evaluation_steps", type=int, default=100, help="Number of steps between checkpoints (saved via save_steps)")
    parser.add_argument("--warmup_epochs", type=float, default=0.1, help="Warmup epochs as a fraction of dataset size (used to compute warmup_steps)")

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--target_modules", type=str, default="c_attn,q_proj,v_proj", help="Comma-separated target modules for LoRA")

    # Checkpoint and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to load a LoRA checkpoint (if applicable)")
    parser.add_argument("--wandb_project", type=str, default="debate-rm-training", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    accelerator = Accelerator()
    trainer = DebateRMTrainer(args, accelerator)
    trainer.train()
    # Ensure final adapter and head params are saved to the same directory
    trainer.save_model()
