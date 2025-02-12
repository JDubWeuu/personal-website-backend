from fastapi import APIRouter, HTTPException, status, Body
from ..models.form import Form
from typing import Annotated

router = APIRouter(prefix="/contact")


@router.get("/send-form", tags=["contact"], status_code=status.HTTP_201_CREATED)
async def sendContactEmail(form: Annotated[Form, Body(title="the form a user submitted to contact me")]):
    
    """
    Add the contents of the form into a database to store it, and then forward an email with the contents of the form to your own email
    """
    
    try:
        return {
            "message": "Successfully sent contact form!"
        }
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to process form request."
        )

