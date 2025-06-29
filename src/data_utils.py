import os
import requests
import zipfile
from pathlib import Path

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