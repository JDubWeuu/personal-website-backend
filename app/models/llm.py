from pydantic import BaseModel
from typing import Optional


class LLMResponse(BaseModel):
    response: str
    link: Optional[str]


class AgentResponse(BaseModel):
    data: LLMResponse
    contact: bool
