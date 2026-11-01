
import asyncio

from proxigen_rag.ingestion.loading_and_chunking import load_and_split_documents
from proxigen_rag.ingestion.embeddings import get_embedding_model
from proxigen_rag.ingestion.sparse_embeddings import get_sparse_embedding_model
from proxigen_rag.ingestion.vector_store import initialize_collection, store_documents


async def ingest_documents():
    # 1. Load and split documents
    documents = load_and_split_documents()

    if not documents:
        print("No documents found to ingest.")
        return

    # 2. Prepare texts
    texts = [doc.page_content for doc in documents]

    # 3. Dense embeddings
    embedding_model = get_embedding_model()
    dense_vectors = await embedding_model.aembed_documents(texts)

    # 4. Sparse embeddings
    sparse_model = get_sparse_embedding_model()
    sparse_vectors = list(sparse_model.embed(texts))

    # 5. Initialize collection and store documents
    await initialize_collection()
    await store_documents(
        documents,
        texts,
        dense_vectors,
        sparse_vectors,
    )

    print(f"Successfully ingested {len(documents)} chunks.")


if __name__ == "__main__":
    asyncio.run(ingest_documents())
