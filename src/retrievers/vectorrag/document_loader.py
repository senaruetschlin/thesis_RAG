from typing import List
from langchain.docstore.document import Document
import json

def load_json_documents(path: str) -> List[Document]:
    with open(path, "r") as f:
        data = json.load(f)

    docs = []
    for idx, item in enumerate(data):
        if isinstance(item, str) and item.strip():
            docs.append(Document(
                page_content=item.strip(),
                metadata={"row_index": idx}
            ))

    return docs