from fastapi import APIRouter, HTTPException, status, Body, BackgroundTasks, Depends
import os
import aiohttp
from aiohttp import ClientResponse, BasicAuth
from ..models.form import Form, FormCaptcha
from typing import Annotated
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from ..database.db import get_db_connection
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/contact")


async def sendEmail(name: str = "", email: str = "", message_content: str = ""):
    """Use some third party service to asyncronously send the email"""

    MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY")
    if not MAILGUN_API_KEY:
        raise ValueError("MAILGUN_API_KEY is not set in the environment variables.")

    mailgun_domain = "sandboxffffaf00d7de4fca92c1a2fa5df419f3.mailgun.org"
    url = f"https://api.mailgun.net/v3/{mailgun_domain}/messages"

    async with aiohttp.ClientSession() as session:
        try:
            response: ClientResponse = await session.post(
                url,
                auth=BasicAuth("api", MAILGUN_API_KEY),
                data={
                    "from": f"Mailgun Sandbox <postmaster@{mailgun_domain}>",
                    "to": "Jason Wu <wu80jason8@gmail.com>",
                    "subject": f"Message from {name} at {email}!",
                    "text": f"This is an email from: {name}\n\nContent: {message_content}",
                },
            )
            response_text = await response.text()

            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Error sending email: {response_text}",
                )

            return {"message": "Email sent successfully", "response": response_text}
        except Exception as e:
            raise e


@router.post("/send-form", tags=["contact"], status_code=status.HTTP_201_CREATED)
async def sendContactEmail(
    form: Annotated[FormCaptcha, Body(title="the form a user submitted to contact me")],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_connection),
):
    """
    Add the contents of the form into a database to store it, and then forward an email with the contents of the form to your own email
    """

    try:
        async with aiohttp.ClientSession() as session:
            response: ClientResponse = await session.post(
                f"https://www.google.com/recaptcha/api/siteverify",
                data={
                    "secret": os.getenv("GOOGLE_CAPTCHA_SECRET_KEY"),
                    "response": form.captchaCode,
                },
            )
            data = await response.json()
            if not data.get("success"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Captcha code is not valid.",
                )
        background_tasks.add_task(sendEmail, form.name, form.email, form.content)
        await db.execute(
            text(
                "INSERT INTO contact_history (name, email, content) VALUES (:name, :email, :content)"
            ),
            {"name": form.name, "content": form.content, "email": form.email},
        )
        await db.commit()
        print("passed")
        return {"message": "Successfully sent contact form!"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email configuration error: " + str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred: " + str(e),
        )
