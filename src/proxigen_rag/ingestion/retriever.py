





from proxigen_rag.ingestion.vector_store import get_vector_store


def get_retriever(
    search_type: str = "similarity",
    k: int = 10,
):
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
        },
    )










# import os

# from dotenv import load_dotenv
# from qdrant_client import models

# from proxigen_rag.ingestion.vector_store import get_qdrant_client

# load_dotenv()


# def get_retriever(k: int = 5):
#     client = get_qdrant_client()

#     def retrieve(dense_vector, sparse_vector):
#         results = client.query_points(
#             collection_name=os.environ["QDRANT_COLLECTION_NAME"],
#             prefetch=[
#                 models.Prefetch(
#                     query=dense_vector,
#                     using="dense",
#                     limit=k,
#                 ),
#                 models.Prefetch(
#                     query=sparse_vector,
#                     using="sparse",
#                     limit=k,
#                 ),
#             ],
#             query=models.FusionQuery(fusion=models.Fusion.RRF),
#             limit=k,
#             with_payload=True,
#         )

#         return results.points

#     return retrieve
