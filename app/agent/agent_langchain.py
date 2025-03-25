from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import StructuredTool
from .agent_tools import get_tools
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os
import asyncio
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


class ResponseModel(BaseModel):
    content: str = Field(
        description="If the user indeed wants to contact me, then make sure to populate populate the 'content' stating that they definitely can and MAKE SURE TO ASK THEM for their 'name', 'email,' and the message they want to send to me."
    )
    relevant: bool = Field(
        description="This should signal whether the user is trying to contact me or not"
    )


class Agent:
    def __init__(self) -> None:
        self.all_tools = [*get_tools()]
        self.tools = self.all_tools[:2]
        self.nav_tool = self.all_tools[-1]
        self.prompt = PromptTemplate(
            template="""
            Answer the following questions as best you can. If the question DOES NOT relate to Jason, respond as such. You have access to the following tools:

{tools}
You can ONLY call each tool ONCE.
If you call the 'retrieval' tool, make sure to pass in the entire user's query to answer their question.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 3 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""",
            input_variables=[
                "tools",
                "tool_names",
                "input",
                "agent_scratchpad",
                "cache",
            ],
        )
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192",
            temperature=0,
        )
        self.checker_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0,
            cache=False,
        )
        # create agent chain
        self.react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )
        self.executor = AgentExecutor(
            agent=self.react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    async def get_cache_data(self, query: str) -> str:
        """Fetch relevant data from cache to include in prompt"""
        try:
            # This will hit the cache if available
            messages = [
                AIMessage(
                    content="You are supposed to answer questions by responding with information about Jason from the redis cache. DO NOT LIE if you don't have any information. DO NOT preface, only provide the information about Jason if you have any."
                ),
                HumanMessage(content=query),
            ]
            cache_result = await self.llm.ainvoke(messages)
            if cache_result.content:
                return cache_result.content

            return "No cache data available."
        except Exception as e:
            print(f"Error getting cache data: {e}")
            return "No cache data available due to error."

    async def call_agent(self, query: str):
        """Run the agent with the query and get navigation information"""
        # Get relevant cache data first
        # cache_data = await self.get_cache_data(query)
        # print(cache_data)
        # Execute agent with query, including cache data
        tasks = []
        tasks.append(self.executor.ainvoke({"input": query}))
        tasks.append(self.nav_tool.ainvoke(query))
        res = await asyncio.gather(*tasks)
        return res

    async def initial_check(self, query: str):
        """Check whether the user is trying to contact me or not"""
        self.checker_llm = self.checker_llm.with_structured_output(schema=ResponseModel)

        # Using proper message formatting
        res: ResponseModel = await self.checker_llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": "You are Jason's assistant. Your job is to determine if users are trying to contact Jason (for work, meetings, etc.) or just asking questions about him.\n\nIf they want to contact Jason:\n- Set 'relevant' to true\n- In the 'content' field, confirm they can contact Jason\n- Ask for their name, email, and the message they want to send\n\nIf they're just asking questions about Jason or other topics:\n- Set 'relevant' to false\n\nAnother system will handle general questions about Jason, so only identify genuine contact requests.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ]
        )

        if res.relevant == False:
            data = await self.call_agent(query=query)
            return {
                "output": data[0]["output"],
                "link": data[1].url,
                "contact": res.relevant,
            }

        return {"output": res.content, "contact": res.relevant}

    # async def close_cache(self):
    #     await self.cache.aclear()


agent = Agent()


async def main():
    # Test a non-contact query first
    # agent.close_cache()
    # print("\n=== Testing a general query about Jason ===")
    # college_query = "Where has Jason gone off to college at?"
    # college_res = await agent.initial_check(college_query)
    # print(f"College query response: {college_res}")

    # Test a contact query
    print("\n=== Testing a contact request query ===")
    contact_query = "I'd like to schedule a meeting with Jason, how can I reach him?"
    contact_res = await agent.initial_check(contact_query)
    print(f"Contact query response: {contact_res}")

    # Close connections
    # await agent.close_cache()


if __name__ == "__main__":
    asyncio.run(main())
