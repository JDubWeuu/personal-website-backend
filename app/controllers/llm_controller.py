from fastapi import Query
from typing import Annotated
from ..agent.main_retrieval import PostgresRAG
from ..models.llm import LLMResponse, AgentResponse
from ..agent.agent_langchain import agent


async def get_llm_response(query: str):
    """
    Handle logic here and then create a route for this
    """
    res = await agent.initial_check(query=query)

    return AgentResponse(
        data=LLMResponse(response=res["output"], link=res.get("link")),
        contact=res["contact"],
    )
