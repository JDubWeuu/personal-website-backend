import os
import asyncio
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_core.documents import Document

# print(os.getenv("LLAMA_CLOUD_API_KEY"))

parser = LlamaParse(
    result_type="markdown",
    api_key="llx-dOqLACCzrQICD0gNHLvHSzAwEnn3IcvIrIW0YHzvSuvjH6iF"
)

async def parse_documents(file_paths: list[str]) -> list[Document]:
    reader = SimpleDirectoryReader(input_files=file_paths, file_extractor={
        ".pdf": parser
    })
    docs = await reader.aload_data()

    return docs

# async def parse_pdf(file_paths):
#     tasks = []
#     for file_path in file_paths:
#         loader.file_path = file_path
#         tasks.append(loader.aload())
#     docs_arr = await asyncio.gather(*tasks)
#     return docs_arr

if __name__ == "__main__":
    res = asyncio.run(parse_documents(["app/assets/Jason_Wu_Resume.pdf"]))
    print(res)
