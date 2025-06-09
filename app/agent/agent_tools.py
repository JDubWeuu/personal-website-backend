from langchain_core.tools import tool
from pydantic import HttpUrl, ValidationError, BaseModel, Field
from typing import Optional
import asyncio
from langchain_core.documents import Document
import json
from langchain.chat_models import init_chat_model
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from .main_retrieval import PostgresRAG
import os
from dotenv import load_dotenv

load_dotenv()

"""Create agent with langchain tools. Implement it asyncronously with running background tasks as well to more quickly parse files into vector db"""


class NavigationToolResponse(BaseModel):
    url: str


class ProjectInformationToolResponse(BaseModel):
    tech_stack: list[str]
    description: Optional[str]


class Link(BaseModel):
    name: str
    description: str
    link: str


class RetrievalResponse(BaseModel):
    docs: list[Document]


class QueryInput(BaseModel):
    query: str = Field(..., description="The user's original query that was inputted")


class projectInformationInput(BaseModel):
    projectName: str = Field(
        ...,
        description="The project name which the user is trying to learn about? Specifically, to call this tool, must pass in 'nezerac' or 'visionairy' not case sensitive.",
    )


# convert this to json and pass it into the prompt template of the llm to let it decide
# button name will be the name property here
# essentially will be a dynamic link based upon what the user inputs
LINK_INFO: list[Link] = [
    Link(
        **{
            "name": "About Page",
            "description": "This essentially provides an overview of my personal information (i.e. where I go for university, what I study, what I like to do, etc.)",
            "link": "http://localhost:3000/about",
        }
    ),
    Link(
        **{
            "name": "Projects Page",
            "description": "Gives practical information into the projects that I've done in the past during my free time or for hackathons.",
            "link": "http://localhost:3000/projects",
        }
    ),
    Link(
        **{
            "name": "Experience Page",
            "description": "Provides a more detailed look into the professional software engineering experiences I've garnered like internships or work.",
            "link": "http://localhost:3000/experience",
        }
    ),
    Link(
        **{
            "name": "Contact Page",
            "description": "A way to email and get in touch with me for any inquiries. If users want to contact me or talk with me, redirect them to this page",
            "link": "http://localhost:3000/contact",
        }
    ),
]

PROJECT_INFORMATION = {
    "nezerac": {
        "description": "Secured 2nd Place in an AI hackathon hosted by AWS and Inrix, competing against 349 participants (36 teams). In just 24 hours, my team developed an AI agent that streamlines the process of sourcing suppliers, analyzing product pricing, and negotiating deals. Built with Next.js, AWS Lambda, Bedrock, DynamoDB, SES, and S3, the agent empowers business owners to focus on running and growing their businesses instead of having to dedicate their time to tedious tasks.",
        "tech_stack": [
            "AWS",
            "Lambda",
            "DynamoDB",
            "AWS S3",
            "Next.js",
            "Oxylabs",
            "Python",
            "TypeScript",
        ],
    },
    "visionary": {
        "description": "Secured the Most Likely to Be a Startup prize at a hackathon with over 330+ participants from across California. I developed a backend using FastAPI, Langchain, Google Cloud Speech-to-Text Models, and Browser Use to automate and ease the booking process of flights for those who are visually impaired.",
        "tech_stack": [
            "FastAPI",
            "LangChain",
            "OpenAI API",
            "Next.js",
            "Google Cloud",
            "Python",
            "TypeScript",
        ],
    },
}


# @tool
async def navigation_tool(query: str) -> str:
    """
    This tool helps support by providing users with a way to find out more about me via a route on my webpage.
    Based upon the user's query, should be able to figure out the route to a webpage to get more information (or even linkedin or github).
    """
    #     qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    #     vector_store = QdrantVectorStore(client=qdrant_client, embedding=JinaEmbeddings(), collection_name="links")
    #     retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={
    #         "k": 3,
    #         "score_threshold": 0.7
    #     })
    #     response: list[Document] = await retriever.ainvoke(query)
    #     for i in range(len(response)):
    #         response[i] = response[i].model_dump_json(include={"metadata", "page_content"}, indent=2)
    #     response = json.dumps(response)
    #     llm = init_chat_model("gemma2-9b-it", model_provider="groq")
    #     structured_res = llm.with_structured_output(NavigationToolResponse)

    #     system_template = """You will be given an JSON string of a couple links with their descriptions. Based upon the user's query, provide the link in JSON format which best answers the user's question. If none of the links provided suffice the query, provide an empty url link.
    # Here are the links: {links}

    # The JSON object must use the schema: {schema}"""

    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", system_template),
    #         ("human", "{query}")
    #     ])
    #     llm = prompt | structured_res

    client = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="deepseek-r1-distill-llama-70b",
        temperature=0.3,
    )
    client = client.with_structured_output(schema=NavigationToolResponse)

    messages = [
        {
            "role": "system",
            "content": """Here are a couple links with their description from my personal website -> {links}
            \n\nBased upon the context information of these links as well as the user's query, 
            please respond with the best link that the user can navigate to for more information on their query. 
            Your response should be a JSON object with a 'url' field containing the chosen link.
            """,
        },
        {"role": "user", "content": query},
    ]

    messages[0]["content"] = messages[0]["content"].format(
        links=json.dumps([entry.model_dump() for entry in LINK_INFO], indent=2)
    )

    response = await client.ainvoke(messages)
    return response


# @tool
def obtain_project_information(projectName: str) -> ProjectInformationToolResponse:
    projectDetails = PROJECT_INFORMATION.get(
        projectName, "No details found on project."
    )
    return ProjectInformationToolResponse(**projectDetails).model_dump_json(indent=2)


# @tool
async def retrieval(query: str) -> RetrievalResponse:
    """
    Call this tool every time. Pass in the specific user's query into this tool. Based upon the user's query, responds with information relevant to the query from a vector database which obtains information about myself.
    """
    db = PostgresRAG()
    results = await db.hybrid_search(query, alpha=0.6)
    await db.close()

    # Convert Pinecone results to Document objects
    documents = []
    for match in results:
        doc = Document(
            page_content=match["metadata"]["content"],
            metadata={
                "section": match["metadata"]["section"],
                "subsection": match["metadata"]["subsection"],
            },
        )
        documents.append(doc)

    return RetrievalResponse(docs=documents)


# possibly create a contact tool to automatically email me from Ja-Google


def create_tool(func, func_name: str, description: str, args_schema) -> StructuredTool:
    return StructuredTool.from_function(
        name=func_name,
        description=description,
        args_schema=args_schema,
        coroutine=func,
    )


def get_tools():
    tool_info = [
        {
            "func": retrieval,
            "func_name": "retrieval",
            "description": """
                ONLY CALL THIS TOOL ONCE.
                Pass in the specific user's query into this tool. Based upon the user's query, responds with information relevant to the query from a vector database which obtains information about myself.
                """,
            "args_schema": QueryInput,
        },
        {
            "func": obtain_project_information,
            "func_name": "obtain_project_information",
            "description": """
                 This tool is ONLY used to obtain project information about Nezerac and Visionairy, no other projects. Pass in the project name and from the project name obtain extra details like the tech stack about the project.
            """,
            "args_schema": projectInformationInput,
        },
        {
            "func": navigation_tool,
            "func_name": "navigation_tool",
            "description": """
                Please call this tool in sequence to any other tool call, because this will help support the original tool call.
                This tool helps support by providing users with a way to find out more about me via a link on Jason's website.
                Based upon the user's query, should be able to figure out the route to a webpage to get more information.
                """,
            "args_schema": QueryInput,
        },
    ]

    tools: list[StructuredTool] = [create_tool(**entry) for entry in tool_info]

    return tools


async def testing():
    tools = get_tools()
    response = await tools[0].ainvoke("What does Jason like to do in his free time?")
    print(response)


if __name__ == "__main__":
    asyncio.run(testing())
    # print(obtainProjectInformation.invoke("nezerac"))
