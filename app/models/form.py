from pydantic import BaseModel, Field


class Form(BaseModel):
    name: str
    email: str
    content: str


class FormCaptcha(Form):
    captchaCode: str
