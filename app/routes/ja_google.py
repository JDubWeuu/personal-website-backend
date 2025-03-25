from fastapi import (
    APIRouter,
    status,
    Query,
    HTTPException,
    Request,
    Response,
    Depends,
    Cookie,
)
from typing import Annotated
from ..controllers.llm_controller import get_llm_response
from ..models.llm import AgentResponse
from ..models.form import Form
from uuid import uuid4
import json
import aiohttp


async def get_or_create_session_id(
    request: Request, response: Response, sessionId: str = Cookie(default=None)
):
    if not sessionId:
        sessionId = str(uuid4())
        response.set_cookie(
            key="sessionId",
            value=sessionId,
            max_age=60 * 60 * 24 * 7,
            httponly=True,
            secure=False,  # set to true in prod
            samesite="lax",
        )
    print(request.cookies.get("sessionId"))
    return sessionId


router = APIRouter(prefix="/ja-google", tags=["ja-google"])

"""
To Do:
- See if you need to use api requests instead of having embeddings locally especially for production (to minimize price)
- Might need a timer to clear the cache every certain amount of time, might use celery for this
- Vercel for hosting frontend
- Add more images for yourself on frontend
- TanStack query to obtain llm output from the backend
- AWS to host backend
"""


@router.get(
    "/query",
    status_code=status.HTTP_200_OK,
    response_model=AgentResponse,
)
async def query(
    question: Annotated[
        str, Query(alias="q", description="User query to find out more about Jason")
    ],
    request: Request,
    sessionId: Annotated[str, Depends(get_or_create_session_id)],
):
    if len(question) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty."
        )
    response = await get_llm_response(question)
    if response.contact == True:
        redis_res = await request.app.state.redis_client.set(
            f"user:{sessionId}", "send_email"
        )
    return response


# call this endpoint after on the frontend seeing that the response from 'query' endpoint was the contact me
@router.post("/send-email", status_code=status.HTTP_201_CREATED)
async def llm_send_email(request: Request, form: Form):
    redis_client = request.app.state.redis_client
    if redis_client.exists(f"user:{request.cookies.get('sessionId')}"):
        async with aiohttp.ClientSession() as client:
            try:
                data = {"name": form.name, "email": form.email, "content": form.content}
                # change this endpoint when you host your api on prod
                response = await client.post(
                    "http://127.0.0.1:8000/contact/send-form", json=data
                )
                if response.status != 201:
                    raise HTTPException(
                        status_code=response.status,
                        detail="Unable to send email, please try again",
                    )
            except HTTPException as e:
                raise e
    await redis_client.delete(f"user:{request.cookies.get("sessionId")}")
    # on the front end you redirect client side back to ja-google main page
    return {"message": "Successfully sent email to Jason."}
