from qdrant_client import AsyncQdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

async def embed_links():
    async_qdrant_client = AsyncQdrantClient(url=os.getenv("QDRANT_URL"))
    await async_qdrant_client.create_collection("links")
    
    await async_qdrant_client.close()