import os
import requests
import zipfile
from pathlib import Path
import re, decimal
import tiktoken

def download_finqa_dataset(output_path="data"):
    """
    Download FinQA dataset from GitHub if not already present.
    
    Args:
        output_path (str or Path): Directory where the dataset should be saved.
                                  Defaults to "data" in the current directory.
    
    Returns:
        str: Path to the extracted dataset directory, or None if download failed.
    """
    # Convert to Path object and create directory if it doesn't exist
    data_dir = Path(output_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists (try both naming conventions)
    finqa_dir_master = data_dir / "FinQA-master"
    finqa_dir_main = data_dir / "FinQA-main"
    
    if finqa_dir_master.exists():
        print("FinQA dataset already exists (FinQA-master), skipping download")
        return str(finqa_dir_master)
    elif finqa_dir_main.exists():
        print("FinQA dataset already exists (FinQA-main), skipping download")
        return str(finqa_dir_main)
    
    # Download URL (using main branch)
    url = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"
    zip_path = data_dir / "finqa_dataset.zip"
    
    try:
        print("Downloading FinQA dataset...")
        print(f"URL: {url}")
        print(f"Output directory: {data_dir}")
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save the file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting dataset...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up the zip file
        zip_path.unlink()
        
        # Check which directory was created
        if finqa_dir_main.exists():
            finqa_dir = finqa_dir_main
        elif finqa_dir_master.exists():
            finqa_dir = finqa_dir_master
        else:
            # List what was actually extracted
            extracted_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("FinQA")]
            if extracted_dirs:
                finqa_dir = extracted_dirs[0]
            else:
                raise Exception("Could not find extracted FinQA directory")
        
        print("FinQA dataset downloaded and extracted successfully!")
        print(f"Dataset location: {finqa_dir}")
        
        return str(finqa_dir)
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return None
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def download_confinqa_dataset(output_path="data"):
    """
    Download ConFinQA dataset from GitHub if not already present.

    Args:
        output_path (str or Path): Directory where the dataset should be saved.
                                   Defaults to "data" in the current directory.

    Returns:
        str: Path to the extracted dataset directory, or None if download failed.
    """
    import requests, zipfile
    from pathlib import Path

    data_dir = Path(output_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check for both possible extracted folder names
    confinqa_dir_main = data_dir / "ConFinQA-main"
    confinqa_dir_master = data_dir / "ConFinQA-master"

    if confinqa_dir_main.exists():
        print("ConFinQA dataset already exists (ConFinQA-main), skipping download")
        return str(confinqa_dir_main)
    elif confinqa_dir_master.exists():
        print("ConFinQA dataset already exists (ConFinQA-master), skipping download")
        return str(confinqa_dir_master)

    url = "https://github.com/czyssrs/ConvFinQA/archive/refs/heads/main.zip"
    zip_path = data_dir / "confinqa_dataset.zip"

    try:
        print("Downloading ConFinQA dataset...")
        print(f"URL: {url}")
        print(f"Output directory: {data_dir}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        zip_path.unlink()

        # Check which directory was created
        if confinqa_dir_main.exists():
            confinqa_dir = confinqa_dir_main
        elif confinqa_dir_master.exists():
            confinqa_dir = confinqa_dir_master
        else:
            extracted_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("ConFinQA")]
            if extracted_dirs:
                confinqa_dir = extracted_dirs[0]
            else:
                raise Exception("Could not find extracted ConFinQA directory")

        print("ConFinQA dataset downloaded and extracted successfully!")
        print(f"Dataset location: {confinqa_dir}")
        return str(confinqa_dir)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return None
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def download_finder_dataset(output_path="data"):
    """
    Download the FinDER dataset from Hugging Face and save it locally.

    Args:
        output_path (str or Path): Directory where the dataset should be saved.
                                   Defaults to "data" in the current directory.

    Returns:
        str: Path to the saved dataset directory.
    """
    from datasets import load_dataset
    from pathlib import Path

    data_dir = Path(output_path) / "FinDER"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the dataset from Hugging Face
    print("Downloading FinDER dataset from Hugging Face...")
    dataset = load_dataset("Linq-AI-Research/FinDER")

    # Save each split as a JSONL file
    for split in dataset.keys():
        split_data = dataset[split]
        out_path = data_dir / f"{split}.jsonl"
        split_data.to_json(str(out_path), orient="records", lines=True)
        print(f"Saved {split} split to {out_path}")

    print(f"FinDER dataset saved in: {data_dir}")
    return str(data_dir)

def preprocess_finqa_sample(sample, tokenizer=None, max_bpe_tokens=100):
    # Set up tokenizer if not provided
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's default

    # Helper: sentence split (simple, can be improved)
    def sentence_split(text):
        import re
        # Split on period, question mark, exclamation, or newline
        return [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n', text) if s.strip()]

    # Helper: concatenate sentences ≤ max_bpe_tokens
    def segment_sentences(sentences):
        segments = []
        current = ""
        for sent in sentences:
            if not current:
                current = sent
            else:
                # Try adding the next sentence
                test = current + " " + sent
                if len(tokenizer.encode(test)) <= max_bpe_tokens:
                    current = test
                else:
                    segments.append(current)
                    current = sent
        if current:
            segments.append(current)
        return segments

    # 1. qid
    qid = "FinQA_" + str(sample["id"])
    # 2. dataset
    dataset = "FinQA"
    # 3. question
    question = sample["qa"]["question"]
    # 4. answer
    answer = sample["qa"]["answer"]
    # 5. context_text
    pre_text = sample.get("pre_text", [])
    post_text = sample.get("post_text", [])
    
    # Handle pre_text and post_text as lists of strings
    if isinstance(pre_text, list):
        pre_sentences = []
        for text_chunk in pre_text:
            if isinstance(text_chunk, str):
                pre_sentences.extend(sentence_split(text_chunk))
    else:
        pre_sentences = sentence_split(pre_text) if isinstance(pre_text, str) else []
    
    if isinstance(post_text, list):
        post_sentences = []
        for text_chunk in post_text:
            if isinstance(text_chunk, str):
                post_sentences.extend(sentence_split(text_chunk))
    else:
        post_sentences = sentence_split(post_text) if isinstance(post_text, str) else []
    
    sentences = pre_sentences + post_sentences
    context_text = segment_sentences(sentences)
    # 6. context_table
    context_table = sample.get("table")
    # 7. reasoning
    reasoning = len(sample["qa"].get("steps", [])) > 1
    # 8. reason_type
    steps = sample["qa"].get("steps", [])
    reason_type = steps[0]["op"] if steps else None
    # 9. gold_text_id
    gold_text_id = ["text_" + str(i) for i in sample["qa"].get("ann_text_rows",[])]
    # 10. gold_table_row
    gold_table_row = sample["qa"].get("ann_table_rows", [])
    # 11. meta
    meta = {
        "tfidftopn": sample.get("tfidftopn"),
        "table_retrieved": sample.get("table_retrieved"),
        "text_retrieved": sample.get("text_retrieved"),
    }

    return {
        "qid": qid,
        "dataset": dataset,
        "question": question,
        "answer": answer,
        "context_text": context_text,
        "context_table": context_table,
        "reasoning": reasoning,
        "reason_type": reason_type,
        "gold_text_id": gold_text_id,
        "gold_table_row": gold_table_row,
        "meta": meta,
    }

# Wrapper to process a whole dataset
def preprocess_finqa_dataset(finqa_data, tokenizer=None, max_bpe_tokens=100):
    return [preprocess_finqa_sample(sample, tokenizer, max_bpe_tokens) for sample in finqa_data]


def canonicalise_answer(ans: str) -> str:
    # remove $ and commas
    ans = ans.replace('$','').replace(',','').strip()
    # convert percentages to decimal strings
    if ans.endswith('%'):
        try:
            ans = str(decimal.Decimal(ans[:-1]) / 100)
        except decimal.InvalidOperation:
            pass
    return ans.lower()


def preprocess_convinqa_sample_simple(sample, tokenizer=None, max_bpe_tokens=100):
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's default

    def sentence_split(text):
        return [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n', text) if s.strip()]

    def segment_sentences(sentences):
        segments = []
        current = ""
        for sent in sentences:
            if not current:
                current = sent
            else:
                test = current + " " + sent
                if len(tokenizer.encode(test)) <= max_bpe_tokens:
                    current = test
                else:
                    segments.append(current)
                    current = sent
        if current:
            segments.append(current)
        return segments

    qid = "ConvFinQA_" + str(sample["id"])
    dataset = "ConvFinQA"
    question = sample["qa"]["question"]
    answer = sample["qa"]["answer"]
    pre_text = sample.get("pre_text", [])
    post_text = sample.get("post_text", [])

    if isinstance(pre_text, list):
        pre_sentences = []
        for text_chunk in pre_text:
            if isinstance(text_chunk, str):
                pre_sentences.extend(sentence_split(text_chunk))
    else:
        pre_sentences = sentence_split(pre_text) if isinstance(pre_text, str) else []

    if isinstance(post_text, list):
        post_sentences = []
        for text_chunk in post_text:
            if isinstance(text_chunk, str):
                post_sentences.extend(sentence_split(text_chunk))
    else:
        post_sentences = sentence_split(post_text) if isinstance(post_text, str) else []

    sentences = pre_sentences + post_sentences
    context_text = segment_sentences(sentences)
    context_table = sample.get("table")
    reasoning = len(sample["qa"].get("steps", [])) > 1
    steps = sample["qa"].get("steps", [])
    reason_type = steps[0]["op"] if steps else None
    gold_text_id = ["text_" + str(i) for i in sample["qa"].get("ann_text_rows", [])]
    gold_table_row = sample["qa"].get("ann_table_rows", [])

    annotation = sample.get("annotation", {})
    # --- Ensure cur_dial is always a list of dicts with a question key ---
    cur_dial = annotation.get("cur_dial")
    if not (isinstance(cur_dial, list) and len(cur_dial) > 0 and isinstance(cur_dial[0], dict)):
        # Fallback: use the current question as the only turn
        cur_dial = [{"question": question}]
    meta = {
        "dialogue_break": annotation.get("dialogue_break", []),
        "turn_ind": annotation.get("turn_ind"),
        "cur_dial": cur_dial,
        "cur_program": annotation.get("cur_program"),
        "cur_type": annotation.get("cur_type"),
        "exe_ans_list": annotation.get("exe_ans_list", []),
        "original_program": annotation.get("original_program"),
        "step_list": annotation.get("step_list", []),
        "answer_list": annotation.get("answer_list", []),
        "qa_split": annotation.get("qa_split"),
        "turn_program": annotation.get("turn_program"),
        "turn_program_ori": annotation.get("turn_program_ori"),
        "dialogue_break_ori": annotation.get("dialogue_break_ori"),
        "gold_ind": annotation.get("gold_ind"),
        "amt_table": annotation.get("amt_table"),
        "amt_pre_text": annotation.get("amt_pre_text"),
        "amt_post_text": annotation.get("amt_post_text"),
    }

    return {
        "qid": qid,
        "dataset": dataset,
        "question": question,
        "answer": answer,
        "context_text": context_text,
        "context_table": context_table,
        "reasoning": reasoning,
        "reason_type": reason_type,
        "gold_text_id": gold_text_id,
        "gold_table_row": gold_table_row,
        "meta": meta,
    }

def preprocess_convinqa_dataset_simple(convinqa_data, tokenizer=None, max_bpe_tokens=100):
    return [preprocess_convinqa_sample_simple(sample, tokenizer, max_bpe_tokens) for sample in convinqa_data]

def preprocess_finder_sample(sample, tokenizer=None, max_bpe_tokens=100):
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Helper: sentence split
    def sentence_split(text):
        return [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n', text) if s.strip()]

    # Helper: concatenate sentences ≤ max_bpe_tokens
    def segment_sentences(sentences):
        segments = []
        current = ""
        for sent in sentences:
            if not current:
                current = sent
            else:
                test = current + " " + sent
                if len(tokenizer.encode(test)) <= max_bpe_tokens:
                    current = test
                else:
                    segments.append(current)
                    current = sent
        if current:
            segments.append(current)
        return segments

    # 1. qid
    qid = "FinDER_" + str(sample["_id"])
    # 2. dataset
    dataset = "FinDER"
    # 3. question
    question = sample["text"]
    # 4. answer (canonicalised)
    answer = canonicalise_answer(sample["answer"])
    # 5. context_text: split each entry in references into sentences or 100-token chunks
    context_text = []
    for ref in sample.get("references", []):
        context_text.extend(segment_sentences(sentence_split(ref)))
    # 6. context_table: always empty
    context_table = []
    # 7. reasoning: already boolean
    reasoning = sample["reasoning"]
    # 8. reason_type: type field
    reason_type = sample.get("type")
    # 9. gold_text_id: ["ref_0", "ref_1", ...]
    gold_text_id = [f"ref_{i}" for i in range(len(sample.get("references", [])))]
    # 10. gold_table_row: always empty
    gold_table_row = []
    # 11. meta: category
    meta = {"category": sample.get("category")}

    return {
        "qid": qid,
        "dataset": dataset,
        "question": question,
        "answer": answer,
        "context_text": context_text,
        "context_table": context_table,
        "reasoning": reasoning,
        "reason_type": reason_type,
        "gold_text_id": gold_text_id,
        "gold_table_row": gold_table_row,
        "meta": meta,
    }

def preprocess_finder_dataset(finder_data, tokenizer=None, max_bpe_tokens=100):
    return [preprocess_finder_sample(sample, tokenizer, max_bpe_tokens) for sample in finder_data]



#______NEW FUNCTIONS____________________________________________________________________________________________
def transform_finqa_dataset(finqa_data):
    """
    Transforms FinQA dataset into the specified format, correctly
    accessing the nested 'program' key for the 'operation' field.

    Args:
        finqa_data (list of dict): The original FinQA data.

    Returns:
        list of dict: Transformed data.
    """
    transformed = []
    for sample in finqa_data:
        # Compose context: pre_text + table + post_text
        pre_text = sample.get("pre_text", "")
        table = sample.get("table", "")
        post_text = sample.get("post_text", "")
        
        if isinstance(table, list):
            table_str = "\n".join(["\t".join(map(str, row)) for row in table])
        else:
            table_str = str(table)
        context = f"{pre_text}\n{table_str}\n{post_text}".strip()

        # Access nested keys within the 'qa' dictionary
        qa_dict = sample.get("qa", {})

        transformed.append({
            "ID": sample.get("filename", ""),
            "question": qa_dict.get("question", ""),
            "answer": qa_dict.get("answer", ""),
            "context": context,
            "gold_context": qa_dict.get("gold_inds", []),
            # This is the corrected line
            "operation": qa_dict.get("program", "")
        })
    return transformed



def transform_convfinqa_dataset(convfinqa_data):
    """
    Transforms FinQA dataset into the specified format, correctly
    accessing the nested 'program' key for the 'operation' field.

    Args:
        finqa_data (list of dict): The original FinQA data.

    Returns:
        list of dict: Transformed data.
    """
    transformed = []
    for sample in convfinqa_data:
        # Compose context: pre_text + table + post_text
        pre_text = sample.get("pre_text", "")
        table = sample.get("table", "")
        post_text = sample.get("post_text", "")
        
        if isinstance(table, list):
            table_str = "\n".join(["\t".join(map(str, row)) for row in table])
        else:
            table_str = str(table)
        context = f"{pre_text}\n{table_str}\n{post_text}".strip()

        # Access nested keys within the 'qa' dictionary
        qa_dict = sample.get("qa", {})
        ann_dict = sample.get("annotation", {})

        transformed.append({
            "ID": sample.get("filename", ""),
            "question": qa_dict.get("question", ""),
            "answer": qa_dict.get("answer", ""),
            "context": context,
            "gold_context": qa_dict.get("gold_inds", []),
            # This is the corrected line
            "operation": qa_dict.get("program", ""),
            # Add turn_ind from annotation
            "turn_ind": ann_dict.get("turn_ind", "")
            })
    return transformed