from fastapi import APIRouter
from ..models.form import Form

router = APIRouter(prefix="/contact")


@router.get("/send-form", tags=["contact"])
def sendContactEmail(form: Form):
    """
    Add the contents of the form into a database to store it, and then forward an email with the contents of the form to your own email
    """

