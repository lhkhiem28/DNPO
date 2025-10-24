lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks arc_challenge  --apply_chat_template --fewshot_as_multiturn --num_fewshot 25
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks truthfulqa_mc2 --apply_chat_template --fewshot_as_multiturn --num_fewshot 0
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks winogrande     --apply_chat_template --fewshot_as_multiturn --num_fewshot 5
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks gsm8k          --apply_chat_template --fewshot_as_multiturn --num_fewshot 5
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks hellaswag      --apply_chat_template --fewshot_as_multiturn --num_fewshot 10
lm_eval --model hf --model_args pretrained=$1 --device cuda:0 --batch_size 16 --tasks mmlu           --apply_chat_template --fewshot_as_multiturn --num_fewshot 5