export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/multi_default.yaml"
export CUDA_VISIBLE_DEVICES="4,5,6,7"
NUM_PROCESSES=4
MASTER_PORT=29500

if [ -z "$1" ]; then
  OUTPUT_DIR="./test"
else
  OUTPUT_DIR="$1"
fi

mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

DATA_DIR=../data/iHarmony4

if [ -z "$2" ]; then
  TEST_FILE=test.jsonl
else
  TEST_FILE="$2"
fi

if [ -z "$3" ]; then
  VAE_FILE=checkpoints/clear_vae
else
  VAE_FILE="$3"
fi

accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/inference/infer.py \
  --pretrained_model_name_or_path checkpoints/stable-diffusion-inpainting \
  --pretrained_vae_model_name_or_path $VAE_FILE \
  --pretrained_unet_model_name_or_path checkpoints/stable-diffusion-inpainting/unet \
  --pretrained_text_encoder_name_or_path checkpoints/stable-diffusion-inpainting/text_encoder \
  --pretrained_controlnet_name_or_path checkpoints/stable-diffusion-inpainting/controlnet \
  --dataset_root $DATA_DIR \
  --test_file $TEST_FILE \
  --output_dir $OUTPUT_DIR \
  --seed=0 \
  --resolution=1024 \
  --output_resolution=256 \
  --eval_batch_size=2 \
  --dataloader_num_workers=8 \
  --mixed_precision="fp16" \
  --use_controller \



python scripts/evaluate/main.py \
	--input_dir $OUTPUT_DIR \
	--output_dir $OUTPUT_DIR-evaluation \
	--data_dir $DATA_DIR \
	--json_file_path $TEST_FILE \
	--resolution=256 \
	--num_processes=512 \
	--use_gt_bg