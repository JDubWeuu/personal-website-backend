from fastapi import FastAPI, Depends, status
from typing import Annotated
from app.routes.contact import router as email_router
from app.routes.ja_google import router as jagoogle_router
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from app.database.db import sessionManager
from app.agent.agent_langchain import agent
from app.database.redis import create_redis_connection
from redis import Redis

load_dotenv()


origins = ["http://localhost:3000", "https://jasonwu.dev", "https://www.jasonwu.dev"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # before yield is what to do when fastapi server first boots
    app.state.redis_client = await create_redis_connection()
    yield
    # clean up operations after api finishes or the server terminates
    if sessionManager._engine is not None:
        await sessionManager.close()
    await app.state.redis_client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(email_router)
app.include_router(jagoogle_router)


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {"message": "API is up and running!"}
