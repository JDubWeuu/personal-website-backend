# from typing_extensions import TypedDict
# from typing import Annotated, Callable, Literal, Optional
# from langgraph.graph import START, END, StateGraph
# from langgraph.types import interrupt, Command
# from langgraph.graph.message import add_messages
# from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
# from langchain_core.documents import Document
# from langchain_core.tools import InjectedToolCallId, tool, StructuredTool
# from pydantic import BaseModel
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.agents import AgentAction, AgentFinish
# from agent_tools import get_tools, RetrievalResponse
# from langchain_groq import ChatGroq
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain import hub
# from IPython.display import Image, display
# import os
# import asyncio
# from dotenv import load_dotenv
# import operator

# load_dotenv()

# """
# Step 1: start node is to conduct retrieval based upon user input
# Step 2: From that output with the documents, it'll send those documents to another node with the user's original input which also stores an LLM binded to a couple tools
# Step 3: If the obtain project info tool is called, then we can add that to LLM context with the documents, and then also redirect to another node which generates a link to find more information about me on my personal website
# Step 4: If the send email tool is called, then we will conduct human-in-the-loop where the llm should ask the user for their contact details (name, email, content they want to send), then we can verify that their email is valid, then go to another node which will then actually perform the email sending process (via api request to my original endpoint or using celery)
# """


# @tool("send_email")
# def send_email():
#     """If user wants to reach out to me or contact me, call this tool."""
#     human_details = interrupt(
#         {
#             "question": "What is your name, email, and also the message you want to provide to Jason? Please put it in the format below\n\nName:\nEmail:\nMessage:\n",
#         }
#     )


# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     documents: list[Document]
#     intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


# tools: list[StructuredTool] = [*get_tools()]
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="deepseek-r1-distill-llama-70b",
#     temperature=0,
# )
# # try and first take away the retrieval tool since we're calling that every time at the start node
# llm_with_tools = llm.bind_tools(tools[1:], tool_choice="any")


# class FinalAnswerInput(BaseModel):
#     state: dict


# @tool("final_answer", args_schema=FinalAnswerInput)
# async def final_answer(state: dict) -> State:
#     """Obtain the final answer after enough information from tools is gathered"""
#     print("RUNNING FINAL ANSWER NODE...")
#     # Get the original question and documents from messages
#     messages = state["messages"]
#     original_question = messages[0].content if messages else ""
#     documents = state["documents"]

#     # Get results from previous tool calls
#     intermediate_steps = state.get("intermediate_steps", [])
#     tool_results = []
#     for step in intermediate_steps:
#         if step.log != "TBD":
#             tool_results.append(f"From {step.tool}: {step.log}")

#     # Format all gathered information
#     context = f"""
#     Original Question: {original_question}

#     Retrieved Documents:
#     {[doc.page_content for doc in documents]}

#     Tool Results:
#     {'\n'.join(tool_results)}
#     """

#     # Use LLM to generate final answer
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that provides clear, concise answers about Jason based on the provided context and tool results.",
#         },
#         {
#             "role": "user",
#             "content": f"Based on this context, please provide a final answer to the question: {context}",
#         },
#     ]

#     result = await llm.ainvoke(messages)

#     # Return state update with AIMessage
#     return {"messages": [AIMessage(content=result.content)]}


# tools.append(final_answer)


# # first node
# async def obtain_relevant_docs(state: State):
#     query = state["messages"][-1].content
#     retrieval: StructuredTool = tools[0]
#     # Create the input in the correct format
#     res: RetrievalResponse = await retrieval.arun(query)

#     state["documents"] = res.docs

#     return {"messages": [HumanMessage(content=query)], "documents": res.docs}


# system_prompt = """
#         You are the oracle, the great AI decision maker specifically for Jason's personal website.
#         Given the user's query about Jason, you must decide what to do with it based on a list of tools provided to you.

#         If you see a that a tool has been used (in the scratchpad) with a particular query, do NOT use that same tool with the same query again.

#         Once you have collected enough information to answer the user's question about Jason (stored in the scratchpad) use the final_answer tool.
#         """

# prompt = ChatPromptTemplate(
#     [
#         ("system", system_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("user", "{input}"),
#         ("assistant", "scratchpad: {scratchpad}"),
#     ]
# )


# def create_scratchpad(intermediate_steps: list[AgentAction]):
#     steps = []
#     for i, action in enumerate(intermediate_steps):
#         if action.log != "TBD":
#             steps.append(
#                 f"Tool: {action.tool}, input: {action.tool_input}\n"
#                 f"Output: {action.log}",
#             )

#     return "\n---\n".join(steps)


# oracle = (
#     {
#         "input": lambda x: x["input"],
#         "chat_history": lambda x: x["chat_history"],
#         "scratchpad": lambda x: create_scratchpad(
#             intermediate_steps=x["intermediate_steps"]
#         ),
#     }
#     | prompt
#     | llm_with_tools
# )


# # second node
# async def llm_node(state: State) -> State:
#     print("RUNNING LLM NODE...")
#     # documents = state["documents"]
#     messages = state["messages"]

#     # Get the original question (should always be the first message)
#     original_question = messages[0].content if messages else ""

#     try:
#         # Format the input for the oracle
#         oracle_input = {
#             "input": original_question,  # Just use the original question since docs are in messages
#             "chat_history": messages,  # Include all messages since they contain the retrieved docs
#             "intermediate_steps": state["intermediate_steps"],
#         }

#         print(oracle_input)

#         result = await oracle.ainvoke(oracle_input)
#         print(result)
#         tool_name = result.tool_calls[0]["name"]
#         tool_args = result.tool_calls[0]["args"]

#         action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")

#         return {
#             **state,
#             "intermediate_steps": state["intermediate_steps"] + [action_out],
#         }
#     except Exception as e:
#         print(f"Error in LLM node: {str(e)}")
#         raise


# def router(state: State):
#     print("ROUTER IS RUNNING...")
#     if not state["intermediate_steps"]:
#         return "final_answer"

#     last_step = state["intermediate_steps"][-1]
#     tool_name = last_step.tool

#     # Only route to final_answer if the LLM explicitly called it
#     if tool_name == "final_answer":
#         return "final_answer"

#     # Otherwise, return the tool name for navigation or project info
#     return tool_name


# async def run_nav_tool(state: State):
#     print("NAV TOOL IS RUNNING...")
#     tool_name = state["intermediate_steps"][-1].tool
#     tool_args = state["intermediate_steps"][-1].tool_input

#     out = await tools[1].ainvoke(input=tool_args)
#     action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))

#     # print(action_out)

#     return {
#         **state,
#         "intermediate_steps": state["intermediate_steps"] + [action_out],
#     }


# async def run_obtain_proj(state: State):
#     print("OBTAIN PROJ TOOL IS RUNNING...")
#     tool_name = state["intermediate_steps"][-1].tool
#     tool_args = state["intermediate_steps"][-1].tool_input

#     out = await tools[2].ainvoke(input=tool_args)
#     action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))

#     return {
#         **state,
#         "intermediate_steps": state["intermediate_steps"] + [action_out],
#     }


# workflow = StateGraph(State)
# workflow.add_node("retrieval", obtain_relevant_docs)
# workflow.add_edge(START, "retrieval")
# workflow.add_node("llm_node", llm_node)
# workflow.add_edge("retrieval", "llm_node")
# workflow.add_node("navigation_tool", run_nav_tool)
# workflow.add_node("obtain_project_information", run_obtain_proj)
# workflow.add_edge("navigation_tool", "llm_node")
# workflow.add_edge("obtain_project_information", "llm_node")
# workflow.add_node("final_answer", final_answer)
# workflow.add_edge("llm_node", "final_answer")
# workflow.add_edge("final_answer", END)

# workflow.add_conditional_edges(source="llm_node", path=router)

# graph = workflow.compile()

# # display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# user_input = {
#     "messages": [
#         HumanMessage(content="Where has Jason interned at?")
#     ],  # Include the initial message
#     "chat_history": [],
#     "intermediate_steps": [],
#     "documents": [],  # Initialize empty documents list
# }


# async def main():
#     res = await graph.ainvoke(user_input)
#     print(res)


# asyncio.run(main())
