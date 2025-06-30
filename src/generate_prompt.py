def generate_rag_prompt(retrieved_chunks, question, query_type="general"):
    """
    Generate a standardized prompt for RAG systems.

    Args:
        retrieved_chunks (str): The retrieved context to supply to the model.
        question (str): The user query.
        query_type (str): The query type (e.g. 'general', 'numeric', 'structure'). 

    Returns:
        str: The formatted prompt.
    """
    
    base_instructions = (
        "You are an expert financial analyst. "
        "Answer the following question strictly using the provided context. "
        "If the answer is not present, say 'Not found in context.'"
    )

    if query_type == "numeric":
        additional_instructions = " Show reasoning steps for numeric questions."
    elif query_type == "structure":
        additional_instructions = " Pay attention to document structure and section headings."
    else:
        additional_instructions = ""

    prompt = (
        f"{base_instructions}{additional_instructions}\n\n"
        f"Context:\n{retrieved_chunks}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )
    
    return prompt


# Example usage
if __name__ == "__main__":
    # Example retrieved context and question
    retrieved_context = (
        "Section 1A: The company identifies cyber security as a key risk factor. "
        "Additionally, supply chain disruptions are mentioned as potential risks."
    )
    
    question = "What are the key risk factors mentioned in section 1A?"
    query_type = "structure"

    # Generate the prompt
    prompt = generate_rag_prompt(retrieved_context, question, query_type)

    # Print the prompt to verify
    print("Generated Prompt:\n")
    print(prompt)