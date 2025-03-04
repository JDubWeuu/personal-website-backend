from langchain_core.tools import tool
from pydantic import HttpUrl, ValidationError, BaseModel
from typing import Annotated
from typing import Optional
from qdrant_client import AsyncQdrantClient
import os

"""Create agent with langchain tools. Implement it asyncronously with running background tasks as well to more quickly parse files into vector db"""

class NavigationToolResponse(BaseModel):
    url: HttpUrl

class ProjectInformationToolResponse(BaseModel):
    techStack: list[str]
    extra_description: Optional[str]
class Link(BaseModel):
    name: str
    description: str
    link: HttpUrl
    

# convert this to json and pass it into the prompt template of the llm to let it decide
# button name will be the name property here
# essentially will be a dynamic link based upon what the user inputs
LINK_INFO: list[Link] = [
    {
        "name": "About Page",
        "description": "Gives detailed information about myself with also my resume on there",
        "link": "http://localhost:3000/about"
    },
    {
        "name": "Projects Page",
        "description": "Gives practical information into the projects that I've done in the past during my free time or for hackathons.",
        "link": "http://localhost:3000/projects"
    },
    {
        "name": "Experience Page",
        "description": "Provides a more detailed look into the professional software engineering experiences I've garnered like internships or work.",
        "link": "http://localhost:3000/experience"
    },
    {
        "name": "Contact Page",
        "description": "A way to email and get in touch with me for any inquiries. If users want to contact me or talk with me, redirect them to this page",
        "link": "http://localhost:3000/contact"
    }
]

PROJECT_INFORMATION = {
    "nezerac": {
        "description": "Secured 2nd Place in an AI hackathon hosted by AWS and Inrix, competing against 349 participants (36 teams). In just 24 hours, my team developed an AI agent that streamlines the process of sourcing suppliers, analyzing product pricing, and negotiating deals. Built with Next.js, AWS Lambda, Bedrock, DynamoDB, SES, and S3, the agent empowers business owners to focus on running and growing their businesses instead of having to dedicate their time to tedious tasks.",
        "tech_stack": ["AWS", "Lambda", "DynamoDB", "AWS S3", "Next.js", "Oxylabs", "Python", "TypeScript"]
    },
    "visionary": {
        "description": "Secured the Most Likely to Be a Startup prize at a hackathon with over 330+ participants from across California. I developed a backend using FastAPI, Langchain, Google Cloud Speech-to-Text Models, and Browser Use to automate and ease the booking process of flights for those who are visually impaired.",
        "tech_stack": ["FastAPI", "LangChain", "OpenAI API", "Next.js", "Google Cloud", "Python", "TypeScript"]
    }
}
@tool(response_format='content_and_artifact')
async def navigation_tool() -> NavigationToolResponse:
    """
    based upon the user's query, should be able to figure out the route to a webpage to get more information (or even linkedin or github)
    """
    

@tool(response_format="content_and_artifact")
def obtainProjectInformation(projectName: str) -> ProjectInformationToolResponse:
    """
    Pass in the project name and from the project name obtain extra details like the tech stack about the project. This tool is only used if the retrieval from vector database cannot yield any firm data on a specific project
    """
    projectDetails = PROJECT_INFORMATION.get(projectName, "No details found on project.")
    return ProjectInformationToolResponse(
        **projectDetails
    ).model_dump_json(indent=2)