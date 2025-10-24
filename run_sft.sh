git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

echo "Running iteration 1"
python gen_data.py --dataset "lhkhiem28/ultrafeedback-iter1" --model "Qwen/Qwen2.5-0.5B-Instruct" --dataset_next "lhkhiem28/ultrafeedback-sft-iter1"
sleep 1m
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py --config recipes/zephyr-7b-beta/sft/config_iter1.yaml
sleep 1m
bash eval.sh lhkhiem28/zephyr-7b-sft-iter1
sleep 1m