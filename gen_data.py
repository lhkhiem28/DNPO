import argparse
import tqdm
from datasets import load_dataset
from datasets import Dataset
from vllm import LLM, SamplingParams
import llm_blender

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset_next", type=str)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)["train_prefs"]
    dataset_next = []

    generator = LLM(model=args.model, tensor_parallel_size=1)
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")

    batch_size = 64
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        prompts, prompts_id = batch["prompt"], batch["prompt_id"]
        outputs = generator.generate(prompts, SamplingParams(
            max_tokens=1024, 
            temperature=0, top_p=1, 
        ))
        for j, output in enumerate(outputs):
            batch["rejected"][j][1]["content"] = output.outputs[0].text

        swaps = blender.compare(prompts, [rejected[1]["content"] for rejected in batch["rejected"]], [chosen[1]["content"] for chosen in batch["chosen"]])
        for j in range(len(batch["prompt"])):
            chosen, rejected = batch["chosen"][j], batch["rejected"][j]
            if swaps[j]:
                chosen, rejected = rejected, chosen

            dataset_next.append({
                "prompt": prompts[j], "prompt_id": prompts_id[j], 
                "chosen": chosen, "rejected": rejected, 
                "swap_preferences": swaps[j], 
            })

    Dataset.from_list(dataset_next).push_to_hub(args.dataset_next)