import json
import pandas as pd

def inspect_missing_gold_references(merged_df_path: str, original_df_path: str, dataset_name: str):
    """
    Finds samples with empty gold references in a merged dataset and inspects
    their corresponding entries in the original dataset.

    Args:
        merged_df_path (str): Path to the merged and preprocessed JSON file.
        original_df_path (str): Path to the original dataset JSON file.
        dataset_name (str): The name of the dataset to inspect (e.g., 'ConvFinQA').

    Returns:
        pd.DataFrame: A DataFrame containing the rows from the original dataset
                      that correspond to the samples with missing gold references.
                      Returns an empty DataFrame if no such samples are found or
                      if files cannot be read.
    """
    try:
        # Load the datasets
        merged_df = pd.read_json(merged_df_path)
        original_df = pd.read_json(original_df_path)
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        return pd.DataFrame()
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return pd.DataFrame()

    # --- Step 1: Find problematic samples in the merged dataframe ---
    # Filter for the specific dataset
    dataset_specific_df = merged_df[merged_df['dataset'] == dataset_name]

    # Helper function to check for empty lists/values
    def is_empty(x):
        return x is None or len(x) == 0

    # Find rows where both gold reference fields are empty
    empty_gold_mask = dataset_specific_df['gold_text_id'].apply(is_empty) & \
                      dataset_specific_df['gold_table_row'].apply(is_empty)
    
    problematic_samples = dataset_specific_df[empty_gold_mask]

    if problematic_samples.empty:
        print(f"No samples with empty gold references found for dataset '{dataset_name}'.")
        return pd.DataFrame()

    # --- Step 2: Get the qids and subset the original dataframe ---
    problematic_qids = problematic_samples['qid'].tolist()
    print(f"Found {len(problematic_qids)} problematic QIDs in '{dataset_name}'.")

    # Ensure the 'qid' column exists in the original dataframe
    if 'qid' not in original_df.columns:
         # FinQA's original data uses 'id' instead of 'qid'
         if 'id' in original_df.columns:
              original_df = original_df.rename(columns={'id': 'qid'})
         else:
              print("Error: 'qid' column not found in the original dataframe.")
              return pd.DataFrame()

    # Subset the original dataframe
    original_subset = original_df[original_df['qid'].isin(problematic_qids)]

    return original_subset

if __name__ == "__main__":
    # Define file paths
    merged_file = "/Users/christel/Desktop/Thesis/thesis_repo/data/data_processed/Train_Val_Test/df_train.json"
    
    # --- Inspect ConvFinQA ---
    print("--- Inspecting ConvFinQA ---")
    original_convfinqa_file = "data/ConvFinQA-main/data/train.json"
    convfinqa_subset = inspect_missing_gold_references(merged_file, original_convfinqa_file, "ConvFinQA")
    
    if not convfinqa_subset.empty:
        print("\nOriginal data for ConvFinQA samples with missing gold references:")
        # Display relevant columns from the original data for inspection
        display_cols_convfinqa = ['qid', 'qa'] # 'qa' contains the gold references in ConvFinQA
        print(convfinqa_subset[display_cols_convfinqa].to_string())

    print("\n" + "="*80 + "\n")

    # --- Inspect FinQA ---
    print("--- Inspecting FinQA ---")
    original_finqa_file = "data/FinQA-main/dataset/train.json"
    finqa_subset = inspect_missing_gold_references(merged_file, original_finqa_file, "FinQA")

    if not finqa_subset.empty:
        print("\nOriginal data for FinQA samples with missing gold references:")
        # Display relevant columns from the original data for inspection
        display_cols_finqa = ['qid', 'qa'] # 'qa' also contains gold references in FinQA
        print(finqa_subset[display_cols_finqa].to_string()) 