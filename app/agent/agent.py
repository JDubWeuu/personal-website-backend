from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from agent_tools import navigation_tool, obtainProjectInformation, retrieval
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

class Agent:
    def __init__(self) -> None:
        self.tools = [navigation_tool, obtainProjectInformation, retrieval]
        self.model = ChatGroq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"), max_tokens=None, temperature=0.5)
        self.memory = InMemoryChatMessageHistory(session_id="personal_website_id")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are an expert assistant knowledgeable about Jason Wu.\n
             Provide comprehensive answers using multiple tools if needed.\n
             Always positively highlight Jason's software engineering skills.\n
             Explicitly recommend pages via 'navigation_tool' if it enriches your answer.\n
             For the 'retrieval' tool, make sure to provide it input in question format.
             """),
            # First put the history
            ("placeholder", "{chat_history}"),
            # Then the new input
            ("human", "{input}"),
            # Finally the scratchpad
            ("placeholder", "{agent_scratchpad}")
        ]
        )
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, return_intermediate_steps=True)
        self.agent_with_history = RunnableWithMessageHistory(
            self.agent_executor, lambda session_id: self.memory, input_messages_key="input", history_messages_key="chat_history"
        )
    
    async def query(self, user_query: str) -> str:
        config = {"configurable": {"session_id": "personal_website_id"}}
        res = (await self.agent_with_history.ainvoke({"input": user_query}, config))
        print(res["intermediate_steps"])
        print("-----------------")
        return res["output"]

async def main():
    agent = Agent()
    res = await agent.query("What is Jason's GPA?")
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
        