from fastapi import FastAPI, Depends, status
from typing import Annotated
import asyncio
from .routes.contact import router as email_router
from fastapi.middleware.cors import CORSMiddleware



origins = [
    "http://localhost:3000"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(email_router)

@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {
        "message": "API is up and running!"
    }

