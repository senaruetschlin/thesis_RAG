# prompt_tuning.py

import json
import os
from typing import List, Dict
import openai

# 1) Prepare your OpenAI key (set as env var or here)
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")

# 2) Convert your merged_dataset.json to a JSONL file for fine-tuning
def prepare_jsonl(
    merged_json_path: str,
    jsonl_out_path: str,
    system_prompt: str,
) -> None:
    """
    Reads your merged_dataset.json and writes a JSONL where each line is:
      {"prompt": "<system_prompt>\nQ: {question}\nA:", "completion": " {answer}"}
    """
    with open(merged_json_path, "r") as f:
        data = json.load(f)

    with open(jsonl_out_path, "w") as outf:
        for sample in data:
            q = sample["question"].strip()
            a = sample["answer"].strip()
            # We add a leading space before completion per OpenAI best practices
            record = {
                "prompt": f"{system_prompt}\n\nQ: {q}\nA:",
                "completion": f" {a}"
            }
            outf.write(json.dumps(record) + "\n")

    print(f"✅ Wrote {len(data)} examples to {jsonl_out_path}")


# 3) Upload and fine‑tune
def fine_tune(jsonl_path: str, base_model="gpt-3.5-turbo", n_epochs=3):
    # 3a) upload
    upload_resp = openai.File.create(
        file=open(jsonl_path, "rb"),
        purpose="fine-tune"
    )
    file_id = upload_resp["id"]
    print(f"Uploaded training file: {file_id}")

    # 3b) start fine‑tuning
    ft_resp = openai.FineTune.create(
        training_file=file_id,
        model=base_model,
        n_epochs=n_epochs,
        batch_size=16,                # tweak as needed
        learning_rate_multiplier=0.1, # often 0.05–0.2
    )
    print("Fine‑tune job started:", ft_resp["id"])
    return ft_resp["id"]


# 4) Poll until done (optional helper)
def wait_for_finetune(ft_id: str, poll_interval_s=30):
    import time
    while True:
        status = openai.FineTune.retrieve(ft_id)["status"]
        print("Status:", status)
        if status in ("succeeded", "failed"):
            break
        time.sleep(poll_interval_s)
    return openai.FineTune.retrieve(ft_id)


# 5) Hook back into your generator
#    After fine‑tune completes, you’ll get `fine_tuned_model` like "ft:gpt-3.5-turbo:…"
#    Pass that into your ChatGPTGenerator constructor.

if __name__ == "__main__":
    # a) set paths
    MERGED = "data_processed/merged_dataset.json"
    JSONL   = "data_processed/finqa_finetune.jsonl"

    # b) decide your new system prompt
    SYSTEM_PROMPT = (
        "You are a highly precise financial assistant.  \n"
        "Use the provided context (text and tables) to answer the question exactly.\n"
        "Show your chain of thought only if asked."
    )

    # c) prepare the file
    prepare_jsonl(MERGED, JSONL, SYSTEM_PROMPT)

    # d) run fine‑tuning
    job_id = fine_tune(JSONL, base_model="gpt-3.5-turbo", n_epochs=4)

    # e) optionally wait until done
    result = wait_for_finetune(job_id)
    print("🎉 Fine‑tune job result:", result)
