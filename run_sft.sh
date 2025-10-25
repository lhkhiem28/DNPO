echo "Running iter1"
python gen_data.py --dataset "lhkhiem28/ultrafeedback-iter1" --model "Qwen/Qwen2.5-3B-Instruct" --dataset_next "lhkhiem28/ultrafeedback-sft-iter1"
sleep 1m
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py --config recipes/zephyr-7b-beta/sft/config_iter1.yaml
sleep 1m
bash eval.sh lhkhiem28/zephyr-7b-sft-iter1
sleep 1m

echo "Running iter2"
python gen_data.py --dataset "lhkhiem28/ultrafeedback-iter2" --model "lhkhiem28/zephyr-7b-sft-iter1" --dataset_next "lhkhiem28/ultrafeedback-sft-iter2"
sleep 1m
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py --config recipes/zephyr-7b-beta/sft/config_iter2.yaml
sleep 1m
bash eval.sh lhkhiem28/zephyr-7b-sft-iter2
sleep 1m

echo "Running iter3"
python gen_data.py --dataset "lhkhiem28/ultrafeedback-iter3" --model "lhkhiem28/zephyr-7b-sft-iter2" --dataset_next "lhkhiem28/ultrafeedback-sft-iter3"
sleep 1m
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml scripts/sft.py --config recipes/zephyr-7b-beta/sft/config_iter3.yaml
sleep 1m
bash eval.sh lhkhiem28/zephyr-7b-sft-iter3
sleep 1m