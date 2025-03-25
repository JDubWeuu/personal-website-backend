import redis.asyncio as redis
import os
from dotenv import load_dotenv

load_dotenv()


async def create_redis_connection():
    connection = redis.from_url(url=os.getenv("REDIS_URL"))

    return connection
