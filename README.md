# Thesis: Comparing VectorRAG and HybridRAG on Financial Documents 

This repository contains the code, notebooks, and data preprocessing scripts for our master's thesis: 
**Comparative Evaluation of Retrieval-Augmented Generation Systems on  Financial
Documents** 

---

## Repository Structure
- **notebooks/** – Analysis and experiments (EDA, VectorRAG, HybridRAG, Evaluation)
- **src/** – Source code for retrievers and generator modules
- **data/** – Raw datasets (not included in repo due to size/privacy)

---

## Setup Instructions
1. Clone the repository:
```bash
   git clone https://github.com/senaruetschlin/thesis_RAG.git
   cd thesis_RAG
```

2. Create and activate a conda environment: 
```bash
   conda env create -f environment.yml
   conda activate thesis_env
```

---

## Requirements
To run the notebooks, you need:

- An **OpenAI API key** with access to the GPT models used.
- A **Neo4j AuraDB** instance (or a local Neo4j instance) with credentials to store and query graph data.

Create a `.env` file in the root directory and include:
```bash
- OPENAI_API_KEY=your_openai_api_key
- NEO4J_URI=your_neo4j_connection_string
- NEO4J_USER=your_neo4j_username
- NEO4J_PASSWORD=your_neo4j_password
```
---

## Usage
Run the Jupyter notebooks in order:
1. `01_eda.ipynb` – Analyze the dataset
2. `01.2_test_data.ipynb` – Prepare the test questions and documents
3. `02_vectorrag.ipynb` – Train and evaluate the VectorRAG pipeline
4. `03_hybridrag.ipynb` – Train and evaluate the HybridRAG pipeline
5. `04_evaluation.ipynb` – Run and compare models and generate final metrics 

---

## Results
| Approach     | Mean nDCG@10 | Micro Precision | Micro Recall | Micro F1 |
|--------------|----------|-----------|--------|----------|
| VectorRAG    | 0.43944      | 0.064865       | 0.530387    | 0.115593      |
| HybridRAG    | 0.586632      | 0.081633       | 0.662983    | 0.145366      |

HybridRAG improves retrieval performance overall compared to VectorRAG, particularly on reasoning-intensive tasks but at the cost of computational efficiency. Retrieval is not solely a preprocessing step but a central pillar of a successful RAG application. Addressing it effectively will be key for researchers and practitioners when scaling LLM adoption in the financial domain. 

---

## License 
This repository is for academic and research purposes. 

---

## Citation
Cochrane, A.; Rütschlin, S. (2025). *Comparative Evaluation of Retrieval-Augmented Generation Systems on Financial Documents.*