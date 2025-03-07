from langchain_core.tools import tool
from pydantic import HttpUrl, ValidationError, BaseModel, Field
from typing import Annotated
from typing import Optional
import asyncio
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.documents import Document
from groq import AsyncGroq
import json
from main_retrieval import PostgresRAG
import os
from dotenv import load_dotenv

load_dotenv()

"""Create agent with langchain tools. Implement it asyncronously with running background tasks as well to more quickly parse files into vector db"""

class NavigationToolResponse(BaseModel):
    url: HttpUrl

class ProjectInformationToolResponse(BaseModel):
    tech_stack: list[str]
    description: Optional[str]
class Link(BaseModel):
    name: str
    description: str
    link: HttpUrl

class RetrievalResponse(BaseModel):
    docs: list[Document]

class QueryInput(BaseModel):
    query: str = Field(..., description="The user's original query that was inputted")

class projectInformationInput(BaseModel):
    projectName: str = Field(..., description="The project name which the user is trying to learn about? Specifically, to call this tool, must pass in 'nezerac' or 'visionairy' not case sensitive.")

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    

# convert this to json and pass it into the prompt template of the llm to let it decide
# button name will be the name property here
# essentially will be a dynamic link based upon what the user inputs
LINK_INFO: list[Link] = [
    {
        "name": "About Page",
        "description": "This essentially provides an overview of my personal information (i.e. where I go for university, what I study, what I like to do, etc.)",
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
@tool(response_format='content', args_schema=QueryInput)
async def navigation_tool(query: str) -> str:
    """
    based upon the user's query, should be able to figure out the route to a webpage to get more information (or even linkedin or github)
    """
    qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    vector_store = QdrantVectorStore(client=qdrant_client, embedding=JinaEmbeddings(), collection_name="links")
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={
        "k": 3,
        "score_threshold": 0.7
    })
    response: list[Document] = await retriever.ainvoke(query)
    for i in range(len(response)):
        response[i] = response[i].model_dump_json(include={"metadata", "page_content"}, indent=2)
    response = json.dumps(response)
    groq_res = await groq_client.chat.completions.create(messages=[
        {
            "role": "system",
            "content": f"You will be given an JSON string of a couple links with their descriptions. Based upon the user's query, provide the link in JSON format which best answers the user's question. If none of the links provided suffice the query, provide an empty url link.\nHere are the links: {response}\n\n"
            f"The JSON object must use the schema: {json.dumps(NavigationToolResponse.model_json_schema(), indent=2)} "
        },
        {
            "role": "user",
            "content": query
        }
    ], model="gemma2-9b-it", temperature=0.1, top_p=1, stream=False, response_format={
        "type": "json_object"
    })
    final_url = json.loads(groq_res.choices[0].to_dict()["message"]["content"])["url"]
    print(final_url)
    return final_url
    

@tool(response_format="content", args_schema=projectInformationInput)
def obtainProjectInformation(projectName: str) -> ProjectInformationToolResponse:
    """
    This is to obtain project information about nezerac and visionairy.
    Pass in the project name and from the project name obtain extra details like the tech stack about the project. This tool is only used if the retrieval tool which obtains information from vector database cannot yield any firm data on a specific project.
    """
    projectDetails = PROJECT_INFORMATION.get(projectName, "No details found on project.")
    return ProjectInformationToolResponse(
        **projectDetails
    ).model_dump_json(indent=2)

@tool(args_schema=QueryInput)
async def retrieval(query: str) -> RetrievalResponse:
    """
    Call this tool every time. Pass in the specific user's query into this tool. Based upon the user's query, responds with information relevant to the query from a vector database which obtains information about myself.
    """
    db = PostgresRAG()
    result = await db.query(query)
    await db.close_connection()
    return RetrievalResponse(docs=result)
    

async def testing():
    response = await navigation_tool.ainvoke("What projects has Jason worked on?")
    print(response)

if __name__ == "__main__":
    # asyncio.run(testing())
    print(obtainProjectInformation.invoke("nezerac"))