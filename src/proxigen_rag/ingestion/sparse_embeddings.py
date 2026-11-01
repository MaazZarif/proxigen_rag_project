
from functools import lru_cache

from fastembed import SparseTextEmbedding

@lru_cache(maxsize=1)
def get_sparse_embedding_model():
    return SparseTextEmbedding(model_name="Qdrant/bm25")
