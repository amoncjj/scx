from datasets import load_dataset
from transformers import AutoTokenizer


subset = "repobench-p"
data = load_dataset("./LongBench", subset, split="test", trust_remote_code=True)

"""
数据格式：
{
    "input": "The input/command for the task, usually short, such as questions in QA, queries in Few-shot tasks, etc",
    "context": "The long context required for the task, such as documents, cross-file code, few-shot examples in Few-shot tasks",
    "answers": "A List of all true answers",
    "length": "Total length of the first three items (counted in characters for Chinese and words for English)",
    "dataset": "The name of the dataset to which this piece of data belongs",
    "language": "The language of this piece of data",
    "all_classes": "All categories in classification tasks, null for non-classification tasks",
    "_id": "Random id for each piece of data"
}
"""

# tokenizer
model_path = "/data/model_hub/DeepSeek-R1-Distill-Llama-70B"
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
eos_id = tok.eos_token_id


def make_prompt(data, isl):
    final_prompt = ""
    for doc in data:
        context = doc["context"]
        query = doc["input"]
        prompt = f"{query}\n{context}"

        final_prompt += prompt

        tokens = tok.encode(final_prompt, add_special_tokens=False)
        if len(tokens) > isl:
            tokens = tokens[:isl]
            final_prompt = tok.decode(tokens)
            break

    return final_prompt


for isl in [1024, 4096, 8192, 32768]:
    prompt = make_prompt(data, isl)
    tokens = tok.encode(prompt, add_special_tokens=False)
    print(len(tokens))

    filename = f"prompt_{isl // 1024}k.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"Saved prompt to {filename}")

        




