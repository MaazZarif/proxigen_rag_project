import time
import requests

from proxigen_rag.ingestion.vector_store import get_vector_store

URL = "http://localhost:6333"
COLLECTION = "proxigen_knowledge"

store = get_vector_store()
query = "What products and services does Proxigen offer?"
vector = store.embeddings.embed_query(query)

endpoint = f"{URL}/collections/{COLLECTION}/points/query"

payload = {
    "query": vector,
    "using": "dense",
    "limit": 5,
    "with_payload": True,
}

session = requests.Session()

for i in range(5):
    start = time.perf_counter()

    response = session.post(endpoint, json=payload, timeout=30)
    response.raise_for_status()

    elapsed = time.perf_counter() - start
    data = response.json()

    print(
        f"Run {i + 1}: "
        f"HTTP={elapsed:.3f}s | "
        f"Qdrant={data.get('time', 'unknown')}s"
    )

session.close()