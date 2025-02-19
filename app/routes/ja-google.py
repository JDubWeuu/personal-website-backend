from fastapi import APIRouter, status, Query
from typing import Annotated

router = APIRouter(prefix="/ja-google")


@router.get("/query", status_code=status.HTTP_200_OK)
def query(question: Annotated[str, Query(title="query to RAG agent", alias="q")]):
    """First check redis if the question has been answered recently, and then provide that answer if it's cached"""
    """Call the RAG agent asyncronously, and then whatever response comes back, store that in a cache on redis, so that if the user tries to use the same query again, it's already stored"""
