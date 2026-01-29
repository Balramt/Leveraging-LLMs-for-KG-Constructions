from __future__ import annotations

import json
import re
import time

import jsonlines
import nltk
import torch
from datasets import Dataset
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.backends.cudnn.benchmark = True

nltk.download("punkt")
nltk.download("wordnet")


def setup_model(model_id: str = "meta-llama/Meta-Llama-3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    return pipe, tokenizer


def load_prompts(filepath: str):
    with jsonlines.open(filepath) as reader:
        return list(reader)


def generate_text(generator, tokenizer, prompts, max_new_tokens: int = 512):
    dataset = Dataset.from_dict({"text": prompts})
    outputs = generator(
        dataset["text"],
        max_new_tokens=max_new_tokens,
        truncation=True,
        num_return_sequences=2,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if isinstance(outputs[0], list):
        outputs = [item for sublist in outputs for item in sublist]
    return outputs


def extract_test_outputs(response):
    outputs = []
    if response and len(response) > 0:
        for res in response:
            generated_text = res.get("generated_text", "")
            match = re.search(r"Test Output:\s*(.*?)(?=\n\s*#|$)", generated_text, re.DOTALL)
            if match:
                outputs.append(match.group(1).strip())
    return outputs if outputs else ["Output not found"]


def parse_model_output(model_output: str):
    triples = []
    lines = [line.strip() for line in model_output.strip().split("\n") if line.strip()]
    pattern = re.compile(r"(.+?)\s*\(([^,]+),\s*([^)]+)\)")
    for line in lines:
        for match in pattern.findall(line):
            relation, subject, obj = match
            triples.append({"sub": subject.strip(), "rel": relation.strip(), "obj": obj.strip()})
    return triples


def save_triples(processed_data, output_filepath: str):
    with open(output_filepath, "w", encoding="utf-8") as outfile:
        for entry in processed_data:
            json.dump({"id": entry["id"], "triples": entry["triples"]}, outfile, ensure_ascii=False)
            outfile.write("\n")


def main(jsonl_path: str, output_path: str, generator, tokenizer, num_prompts: int = 548, batch_size: int = 16):
    prompts = load_prompts(jsonl_path)
    results = []

    for i in range(0, min(num_prompts, len(prompts)), batch_size):
        batch = prompts[i : i + batch_size]
        batch_ids = [item["id"] for item in batch]
        batch_prompts = [item["prompt"] for item in batch]

        try:
            start_time = time.time()
            responses = generate_text(generator, tokenizer, batch_prompts)
            elapsed = time.time() - start_time
            print(f"Batch {i}-{i+len(batch)-1} inference time: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"❌ Error in batch {i}-{i+batch_size}: {e}")
            continue

        for idx in range(len(batch)):
            prompt_responses = responses[2 * idx : 2 * idx + 2]

            test_outputs = extract_test_outputs(prompt_responses)

            all_triples = []
            seen = set()
            for test_output in test_outputs:
                triples = parse_model_output(test_output)
                for triple in triples:
                    triple_key = (triple["sub"], triple["rel"], triple["obj"])
                    if triple_key not in seen:
                        seen.add(triple_key)
                        all_triples.append(triple)

            print(f"[{i + idx + 1}/{num_prompts}] ID: {batch_ids[idx]} → Unique triples extracted: {len(all_triples)}")

            results.append({"id": batch_ids[idx], "triples": all_triples})

    save_triples(results, output_path)
    print(f"\n✅ All {len(results)} prompts processed. Results saved to: {output_path}")


if __name__ == "__main__":
    input_file = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/input/wikidata/improved_input_prompts/"
    output_file = "/upb/users/b/balram/profiles/unix/cs/Text2KG_exp1_thesis/data/output/wikidata/llm_responses/Llama/"
    text_pipe, tokenizer = setup_model("meta-llama/Meta-Llama-3-8B")
    main(input_file, output_file, text_pipe, tokenizer, num_prompts=1500, batch_size=4)
