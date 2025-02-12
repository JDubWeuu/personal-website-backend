from fastapi import FastAPI, Depends, status
from typing import Annotated
import asyncio
from .routes.contact import router as email_router

app = FastAPI()

app.include_router(email_router)

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {
        "message": "API is up and running!"
    }

