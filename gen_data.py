import argparse
import tqdm
from datasets import load_dataset
from datasets import Dataset
from vllm import LLM, SamplingParams

llm_judge_prompt_template = """
You are tasked with evaluating the quality of the given answer based on the provided question. \
Your task is to assign a score between 1 and 100, where 1 indicates very poor quality and 100 indicates excellent quality. \
You should use a 1-point increment scale, meaning the score can be any whole number between 1 and 100 and avoiding scores that are always multiples of 5. \
Consider factors such as relevance, clarity, completeness, and correctness. Provide only the score without any explanation. 
Question: [question]
Answer: [answer]
Score: 
"""

def llm_judge(judge_prompt):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": judge_prompt}
        ],
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int)
    args = parser.parse_args()

    dataset = load_dataset(
        f"lhkhiem28/ultrafeedback-DNPO-iter{args.iter}"
    )["train_prefs"]
    dataset_next = []

    generator = LLM(model=f"lhkhiem28/DNPO-iter{args.iter}", tensor_parallel_size=1)

    batch_size = 32
    for i in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        prompts, prompts_id = batch["prompt"], batch["prompt_id"]
        outputs = generator.generate(prompts, SamplingParams(
            max_tokens=1024, 
            temperature=0, top_p=1, 
        ))
        for j, output in enumerate(outputs):
            batch["rejected"][j][1]["content"] = output.outputs[0].text

        for j in range(len(batch["prompt"])):
            swap = False
            prompt, prompt_id = prompts[j], prompts_id[j]
            chosen, rejected = batch["chosen"][j], batch["rejected"][j]

            chosen_judge_prompt, rejected_judge_prompt = llm_judge_prompt_template.replace("[question]", prompt).replace("[answer]", chosen[1]["content"]), llm_judge_prompt_template.replace("[question]", prompt).replace("[answer]", rejected[1]["content"])
            chosen_score, rejected_score = llm_judge(chosen_judge_prompt), llm_judge(rejected_judge_prompt)
            try:
                chosen_score, rejected_score = float(chosen_score), float(rejected_score)
                if rejected_score > chosen_score:
                    swap = True
                    chosen, rejected = rejected, chosen
                    chosen_score, rejected_score = rejected_score, chosen_score
            except:
                chosen_score, rejected_score = -1, -1

            dataset_next.append({
                "prompt": prompt, "prompt_id": prompt_id, 
                "chosen": chosen, "rejected": rejected, 
                "chosen_score": chosen_score, "rejected_score": rejected_score, 
                "swap_preferences": swap, 
            })

        Dataset.from_list(dataset_next).push_to_hub(
            f"lhkhiem28/ultrafeedback-DNPO-iter{args.iter+1}"
        )