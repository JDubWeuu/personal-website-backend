from fastapi import Query
from typing import Annotated
from ..agent.main_retrieval import PostgresRAG
from ..models.llm import LLMResponse


async def get_llm_response(query: Annotated[str, Query(description="User query to find out more about Jason")]):
    """
    Handle logic here and then create a route for this
    """
    return LLMResponse(query="", response="")