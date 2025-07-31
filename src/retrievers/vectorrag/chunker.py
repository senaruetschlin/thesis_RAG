# src/retrievers/vectorrag/chunker.py
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

def chunk_documents(docs: list[Document], chunk_size=1500, chunk_overlap=200) -> list[Document]:
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = []
    for doc in docs:
        row_index = doc.metadata["row_index"]
        chunks = splitter.split_documents([doc])
        for local_idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"row_{row_index}_chunk_{local_idx}"
            chunked_docs.append(chunk)

    return chunked_docs