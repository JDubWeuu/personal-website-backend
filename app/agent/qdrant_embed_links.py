# from qdrant_client import QdrantClient, AsyncQdrantClient
# import os
# from dotenv import load_dotenv
# from agent_tools import LINK_INFO
# import asyncio
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client.models import VectorParams, Distance
# from langchain_core.documents import Document
# from langchain_community.embeddings import JinaEmbeddings


# load_dotenv()

# qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
# async_qdrant_client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# async def embed_links():
#     if await async_qdrant_client.collection_exists("links"):
#         await async_qdrant_client.delete_collection("links")
#     await async_qdrant_client.create_collection("links", vectors_config=VectorParams(
#                 size=768, distance=Distance.COSINE
#             ))
#     vector_store = QdrantVectorStore(client=qdrant_client, embedding=JinaEmbeddings(), collection_name="links")
#     docs = []
#     for link in LINK_INFO:
#         content = link["name"] + ": " + link["description"]
#         doc = Document(page_content=content, metadata={
#             "link": link["link"]
#         })
#         docs.append(doc)
#     await vector_store.aadd_documents(docs)
#     # res = await vector_store.amax_marginal_relevance_search("What does Jason like to do?", k=2)
#     res = await vector_store.asimilarity_search_with_relevance_scores("What does Jason like to do?", k=2)
#     print(res)
#     await async_qdrant_client.close()
#     # await qdrant_client.close()

# if __name__ == "__main__":
#     asyncio.run(embed_links())
