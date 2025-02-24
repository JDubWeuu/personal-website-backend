from langchain_core.tools import tool
from pydantic import HttpUrl, ValidationError, BaseModel
from typing import Annotated
import json

"""Create agent with langchain tools. Implement it asyncronously with running background tasks as well to more quickly parse files into vector db"""

class NavigationToolResponse(BaseModel):
    url: HttpUrl

class Link(BaseModel):
    name: str
    description: str
    link: HttpUrl
    

# convert this to json and pass it into the prompt template of the llm to let it decide
# button name will be the name property here
# essentially will be a dynamic link based upon what the user inputs
LINK_INFO: list[Link] = [
    {
        "name": "About",
        "description": "Gives detailed information about myself with also my resume on there",
        "link": "http://localhost:3000/about"
    },
    {
        "name": "Projects",
        "description": "Gives practical information into the projects that I've done",
        "link": "http://localhost:3000/projects"
    },
    {
        "name": "Experience",
        "description": "Provides a more detailed look into the professional software engineering experiences I've garnered like internships",
        "link": "http://localhost:3000/experience"
    },
    {
        "name": "Contact",
        "description": "A way to email and get in touch with me for any inquiries. If users want to contact me or talk with me, redirect them to this page",
        "link": "http://localhost:3000/contact"
    }
]

@tool(response_format='content')
def navigation_tool(query: Annotated[str, "The user's query"]) -> NavigationToolResponse:
    """
    based upon the user's query, should be able to figure out the route to a webpage to get more information (or even linkedin or github)
    """
    
    
    
    
    

