from pydantic import BaseModel


class LLMResponse(BaseModel):
    query: str
    response: str