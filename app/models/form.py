from pydantic import BaseModel, Field

class Form(BaseModel):
    name: str = Field(pattern="^[A-Z][a-z]+(?: [A-Z][a-z]+)*(?: [A-Z][a-z]+)?$")
    content: str