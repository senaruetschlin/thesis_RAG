from dotenv import load_dotenv
import os
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"


class ChatGPTGenerator:
    def __init__(self):
        load_dotenv()
        api_key = openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        
        self.client = OpenAI(api_key=api_key, base_url = openai_api_base)


    def generate(self, question, retrieved_docs, system_prompt=None, max_tokens=4000):
        context = "\n".join(retrieved_docs)
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": f"Question: {question}\nContext:\n{context}\nAnswer:"})

        response = self.client.chat.completions.create(
            model="Fin-R1",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            extra_body={"repetition_penalty": 1.05},
        )

        return response.choices[0].message.content.strip()

# Example usage
if __name__ == "__main__":
    generator = ChatGPTGenerator()
    
    question = "What was the net income in Q4 2023?"
    retrieved_docs = [
        "According to the earnings report, the net income in Q4 2023 was $5 million.",
        "Further details are provided in the financial statements."
    ]

    # Try to read the elaborate prompt from file
    try:
        with open("generator_prompt.txt", "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "You are a precise financial assistant. Base answers strictly on context provided."

    print("\nGenerating answer...")
    answer = generator.generate(question, retrieved_docs, system_prompt=system_prompt)
    print("\nGenerated Answer:\n", answer)










# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# class FinR1Generator:
#     def __init__(self, model_name="SUFE-AIFLM-Lab/Fin-R1"):
#         """
#         Initialize Fin-R1 generator on MacBook Pro (M1 Pro chip).
#         """
#         # Determine device: use MPS if available, else CPU
#         if torch.backends.mps.is_available():
#             self.device = "mps"
#         else:
#             self.device = "cpu"
        
#         print(f"Loading Fin-R1 on device: {self.device} ...")
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
#         # Load model (use float32 because M1 doesn’t support bfloat16 or float16 well)
#         self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device)
        
#         print("Fin-R1 model loaded successfully.")

#     def generate(self, question, retrieved_docs, max_new_tokens=200):
#         """
#         Generate an answer using Fin-R1.
        
#         Args:
#             question (str): The input question.
#             retrieved_docs (list[str]): List of retrieved document strings.
#             max_new_tokens (int): Max tokens to generate.

#         Returns:
#             str: Generated text.
#         """
#         context = " ".join(retrieved_docs)
#         prompt = f"Question: {question}\nContext: {context}\nAnswer:"

#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 temperature=0.7,
#                 top_p=0.95,
#                 do_sample=True
#             )

#         output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return output_text
    
# if __name__ == "__main__":
#     generator = FinR1Generator()
    
#     question = "What was the net income in Q4 2023?"
#     retrieved_docs = [
#         "The net income in Q4 2023 was $5 million according to the earnings report.",
#         "Additional details can be found in the financial summary section."
#     ]
    
#     print("\nGenerating answer...")
#     answer = generator.generate(question, retrieved_docs)
#     print("\nGenerated Answer:\n", answer)