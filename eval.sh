lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks arc_challenge  --num_fewshot 25
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks truthfulqa_mc2 --num_fewshot 0
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks winogrande     --num_fewshot 5
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks gsm8k          --num_fewshot 5
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks hellaswag      --num_fewshot 10
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks mmlu           --num_fewshot 5