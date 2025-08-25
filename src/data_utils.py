import os
import requests
import zipfile
from pathlib import Path
import re
import unicodedata

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


#______Preprocessing FUNCTIONS____________________________________________________________________________________________

def _normalize_text(s: str) -> str:
    """Lowercase, unicode-normalize, collapse whitespace & spaces around punctuation."""
    if s is None:
        return ""
    # Unicode compatibility (e.g., non-breaking spaces)
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower()
    # Collapse all whitespace to single space
    s = re.sub(r"\s+", " ", s)
    # Remove spaces before punctuation like " , . ; : ) ] } %"
    s = re.sub(r"\s+([,.;:)\]\}%])", r"\1", s)
    # Remove spaces after opening punctuation like "( [ { "
    s = re.sub(r"([(\[\{])\s+", r"\1", s)
    return s.strip()

def _to_text(x):
    """Safely convert pre/post text that may be list or string to a single string."""
    if isinstance(x, list):
        return " ".join(map(str, x))
    return "" if x is None else str(x)

def _table_to_text(tbl):
    """Convert table (list of rows) to a tab/newline text block, else stringify."""
    if isinstance(tbl, list):
        # if it's a list of lists, join rows with tabs; if flat list, join by newline
        if len(tbl) > 0 and isinstance(tbl[0], list):
            return "\n".join("\t".join(map(str, row)) for row in tbl)
        return "\n".join(map(str, tbl))
    return "" if tbl is None else str(tbl)

def transform_finqa_dataset(finqa_data):
    """
    Transforms FinQA dataset into the specified format,
    keeping gold_context as a dictionary but verifying that
    all *text* gold evidence snippets are included in the context
    (after normalization). Prints a warning if a text snippet
    (keys starting with 'text_') is not found.
    """
    transformed = []
    for sample in finqa_data:
        # Compose context: pre_text + table + post_text (robust to list/str)
        pre_text = _to_text(sample.get("pre_text", ""))
        table_str = _table_to_text(sample.get("table", ""))
        post_text = _to_text(sample.get("post_text", ""))
        context = f"{pre_text}\n{table_str}\n{post_text}".strip()

        # Access nested QA dict
        qa_dict = sample.get("qa", {}) or {}
        gold_dict = qa_dict.get("gold_inds", {}) or {}

        # --- SAFETY CHECK with normalization (only for text_* keys). table_ not an excerpt from the table but a phrase with the evidence---
        norm_context = _normalize_text(context)
        if isinstance(gold_dict, dict):
            for key, gold_text in gold_dict.items():
                if key.startswith("text_"):
                    norm_gold = _normalize_text(str(gold_text))
                    if norm_gold and norm_gold not in norm_context:
                        print(
                            f"WARNING: Text gold snippet from '{sample.get('filename','')}' "
                            f"key '{key}' not found in context. Snippet: '{gold_text[:80]}...'"
                        )

        transformed.append({
            "ID": sample.get("filename", ""),
            "question": qa_dict.get("question", ""),
            "answer": qa_dict.get("answer", ""),
            "context": context,
            "gold_context": gold_dict,           
            "operation": qa_dict.get("program", "")
        })

    return transformed

def transform_convfinqa_dataset(convfinqa_data):
    """
    Transforms ConvFinQA dataset into a flat format where the 'question' field
    includes the full conversation history (Q+A) followed by the current question.
    Adds a safety check to ensure gold text snippets are present in context
    (ignoring table rows).
    """
    transformed = []

    for sample in convfinqa_data:
        # Build the full context (pre_text + table + post_text)
        pre_text = _to_text(sample.get("pre_text", ""))
        table_str = _table_to_text(sample.get("table", ""))
        post_text = _to_text(sample.get("post_text", ""))
        context = f"{pre_text}\n{table_str}\n{post_text}".strip()

        # Extract fields
        qa_dict  = sample.get("qa", {}) or {}
        ann_dict = sample.get("annotation", {}) or {}

        turn_ind = ann_dict.get("turn_ind", 0)
        dialogue = ann_dict.get("dialogue_break", [])
        answers  = ann_dict.get("answer_list", [])

        # Build conversation history up to current turn
        question_turns = []
        for i in range(turn_ind):
            question_turns.append(f"Q: {dialogue[i]}")
            if i < len(answers):
                question_turns.append(f"A: {answers[i]}")
        if turn_ind < len(dialogue):
            question_turns.append(f"Q: {dialogue[turn_ind]}")

        full_question = "\n".join(question_turns)

        # Gold evidence for current turn
        gold_dict = qa_dict.get("gold_inds", {}) or {}

        # --- SAFETY CHECK: verify gold text snippets are present in context ---
        norm_context = _normalize_text(context)
        if isinstance(gold_dict, dict):
            for key, gold_text in gold_dict.items():
                if key.startswith("text_"):  # check only text snippets
                    norm_gold = _normalize_text(str(gold_text))
                    if norm_gold and norm_gold not in norm_context:
                        print(
                            f"WARNING: Gold text snippet from '{sample.get('filename','')}' "
                            f"(turn {turn_ind}) not found in context. Snippet: '{gold_text[:80]}...'"
                        )

        # Build transformed record
        transformed.append({
            "ID": sample.get("filename", ""),
            "question": full_question,
            "answer": qa_dict.get("answer", ""),
            "context": context,
            "gold_context": gold_dict,  
            "operation": qa_dict.get("program", ""),
            "turn_ind": turn_ind
        })

    return transformed

def transform_finder_dataset(finder_data):
    """
    Transforms FinDER dataset into a unified format for RAG evaluation.

    Args:
        finder_data (list of dict): The original FinDER dataset.

    Returns:
        list of dict: Transformed dataset.
    """
    transformed = []
    for sample in finder_data:
        transformed.append({
            "ID": sample.get("_id", ""),
            "question": sample.get("text", ""),
            "answer": sample.get("answer", ""),
            "context": sample.get("references", ""),
            "gold_context": "",  # Empty as FinDER has no gold context
            "operation": sample.get("type", "")
        })
    return transformed