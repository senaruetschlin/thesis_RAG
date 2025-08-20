from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()
# --- For Fin-R1 (vllm local server) ---
# openai_api_key = "EMPTY"
# openai_api_base = "http://0.0.0.0:8000/v1"
# model_name = "Fin-R1"

# --- For OpenAI API (cloud) ---
openai_api_key = os.getenv("OPENAI_API_KEY")  
openai_api_base = "https://api.openai.com/v1"
model_name = "o3-2025-04-16"  #o3 o3-2025-04-16

class ChatGPTGenerator:
    def __init__(self, prompt_path="/Users/christel/Desktop/Thesis/thesis_repo/src/generator/generator_prompt.txt"):
        api_key = openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        self.client = OpenAI(api_key=api_key, base_url=openai_api_base)

        # Load the system prompt from the provided file path
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file not found at: {prompt_path}")

    def generate(self, question, retrieved_docs, max_tokens=4000):
        context = "\n".join(retrieved_docs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}\nContext:\n{context}\nAnswer:"}
        ]

        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content.strip()


# Example usage
#if __name__ == "__main__":
#    generator = ChatGPTGenerator()
#    question = "What was the net income in Q4 2023?"
#    retrieved_docs = [
#        "According to the earnings report, the net income in Q4 2023 was $5 million.",
#        "Further details are provided in the financial statements."
#    ]
#    try:
#        with open("generator_prompt.txt", "r") as f:
#            system_prompt = f.read()
#    except FileNotFoundError:
#        system_prompt = "You are a precise financial assistant. Base answers strictly on context provided."
#
#    print("\nGenerating answer...")
#    answer = generator.generate(question, retrieved_docs, system_prompt=system_prompt)
#    print("\nGenerated Answer:\n", answer)

# --- Fin-R1 (vllm) code for future use ---
# class ChatGPTGenerator:
#     def __init__(self):
#         load_dotenv()
#         api_key = openai_api_key
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY is not set in the environment.")
#         self.client = OpenAI(api_key=api_key, base_url=openai_api_base)
#
#     def generate(self, question, retrieved_docs, system_prompt=None, max_tokens=4000):
#         context = "\n".join(retrieved_docs)
#         messages = []
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})
#         messages.append({"role": "user", "content": f"Question: {question}\nContext:\n{context}\nAnswer:"})
#
#         response = self.client.chat.completions.create(
#             model="Fin-R1",
#             messages=messages,
#             max_tokens=max_tokens,
#             temperature=0.7,
#             top_p=0.95,
#             extra_body={
#                 "repetition_penalty": 1.05,
#             },
#         )
#         return response.choices[0].message.content.strip()