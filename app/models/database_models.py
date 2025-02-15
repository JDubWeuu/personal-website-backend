from pydantic import BaseModel, Field
from sqlalchemy import String, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# database schema
class ContactHistory(Base):
    __tablename__ = 'contact_history'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    content = Column(Text)

# body input model
class ContactCreate(BaseModel):
    name: str
    content: str
    
    
    