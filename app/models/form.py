from pydantic import BaseModel, Field


class Form(BaseModel):
    name: str = Field(pattern="^[A-Z][a-z]+(?: [A-Z][a-z]+)*(?: [A-Z][a-z]+)?$")
    email: str  # I'm already using zod on the frontend to validate the email address
    content: str
