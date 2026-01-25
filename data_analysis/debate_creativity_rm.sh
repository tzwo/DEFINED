export MODEL_PATH="../model/Qwen2.5-7B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file ./accelerate_config_debate_rm.yaml \
  --main_process_port 29501 \
  debate_creativity_rm.py \
  --model_name $MODEL_PATH \
  --learning_rate 4e-4 \
  --max_length 8192 \
  --train_batch_size 2 \
  --val_batch_size 8 \
  --accumulation_steps 8 \
  --num_epochs 30 \
  --evaluation_steps 30 \
  --reward_data_path  \
  --template_path  \
  --checkpoint_dir ../save/test \
  --val_path \
  --oversample_k 1
