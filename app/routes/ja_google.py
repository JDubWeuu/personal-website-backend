from fastapi import (
    APIRouter,
    status,
    Query,
    HTTPException,
    Request,
    Response,
    Depends,
    Cookie,
    BackgroundTasks,
    Body,
)
from typing import Annotated
from ..controllers.llm_controller import get_llm_response
from ..models.llm import AgentResponse
from ..models.form import Form
from uuid import uuid4
import json
import aiohttp
from .contact import sendEmail
from ..database.db import get_db_connection
import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from ..database.dynamodb_integration import dynamo_client


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
# fix this endpoint, because we added the captcha
@router.post("/send-email", status_code=status.HTTP_201_CREATED)
async def llm_send_email(
    request: Request,
    form: Annotated[Form, Body(..., description="form object")],
    backgroundTasks: BackgroundTasks,
    # db: AsyncSession = Depends(get_db_connection),
):
    redis_client = request.app.state.redis_client
    if redis_client.exists(f"user:{request.cookies.get('sessionId')}"):
        backgroundTasks.add_task(sendEmail, form.name, form.email, form.content)
        item = {
            "id": {"S": str(uuid4())},
            "name": {"S": form.name},
            "email": {"S": form.email},
            "content": {"S": form.content},
            "timestamp": {"S": datetime.datetime.now().isoformat()},
        }
        dynamo_client.put_item(TableName="contact", Item=item)
        # await db.execute(
        #     text(
        #         "INSERT INTO contact_history (name, email, content) VALUES (:name, :email, :content)"
        #     ),
        #     {"name": form.name, "content": form.content, "email": form.email},
        # )
        # await db.commit()
        # async with aiohttp.ClientSession() as client:
        #     try:
        #         data = {"name": form.name, "email": form.email, "content": form.content}
        #         # change this endpoint when you host your api on prod
        #         response = await client.post(
        #             "http://127.0.0.1:8000/contact/send-form", json=data
        #         )
        #         if response.status != 201:
        #             raise HTTPException(
        #                 status_code=response.status,
        #                 detail="Unable to send email, please try again",
        #             )
        #     except HTTPException as e:
        #         raise e
    keys_removed = await redis_client.delete(f"user:{request.cookies.get('sessionId')}")
    print(keys_removed)
    # on the front end you redirect client side back to ja-google main page
    return {"message": "Successfully sent email to Jason."}
