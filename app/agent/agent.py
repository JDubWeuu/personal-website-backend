from langchain_core.tools import StructuredTool

"""Create agent with langchain tools. Implement it asyncronously with running background tasks as well to more quickly parse files into vector db"""

def navigation_tool(query: str):
    url_options = {
        "About": {
            "Description": "This page of my site describes why I like software development and building apps. Furthermore, it also talks about the skills I have within software (languages, frameworks, developer tools)."
        },
        "Resume": {
            "Description": "This is just a link to my resume and previous experiences I've been a part of.",
            "Metadata": "This should only be "
        }
    }
    

