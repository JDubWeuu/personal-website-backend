# import os
# import asyncio
# from llama_cloud_services import LlamaParse
# from llama_index.core import SimpleDirectoryReader
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# from langchain_milvus import Milvus, BM25BuiltInFunction
# from uuid import uuid4
# from langchain_nomic import NomicEmbeddings
# from langchain_community.embeddings import JinaEmbeddings
# import nomic
# from langchain.text_splitter import MarkdownHeaderTextSplitter
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# load_dotenv()

# # Initialize external services
# parser = LlamaParse(
#     result_type="markdown",
#     api_key=os.getenv("LLAMA_CLOUD_API_KEY")
# )
# # nomic.login(os.getenv("NOMIC_API_KEY"))

# # Initialize the embedding model
# # nomic_embeddings = NomicEmbeddings(
# #     model="nomic-embed-text-v1.5"
# # )

# jina_embeddings = JinaEmbeddings(
#     api_key=os.getenv("JINA_API_KEY")
# )

# sparse_index_param = {
#     "metric_type": "BM25",
#     "index_type": "AUTOINDEX",
# }

# dense_index_param = {
#     "index_type": "HNSW",
#     "metric_type": "IP"
# }

# URI = "http://localhost:19530"

# # implement a hybrid search
# vector_store = Milvus(embedding_function=jina_embeddings, connection_args={
#         "uri": URI
#     }, collection_name="jason_rag_collection", builtin_function=BM25BuiltInFunction(), vector_field=["dense", "sparse"], consistency_level="Strong", drop_old=True, index_params=[dense_index_param, sparse_index_param])


# async def parse_documents(file_paths: list[str]) -> list[Document]:
#     reader = SimpleDirectoryReader(input_files=file_paths, file_extractor={".pdf": parser})
#     docs = await reader.aload_data()
#     return docs


# def chunk_documents(markdown: str) -> list[Document]:
#     splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Section"), ("##", "Subsection")], strip_headers=False)
#     new_docs = splitter.split_text(markdown)
#     print("The length of the new docs is", len(new_docs))
#     if new_docs:
#         new_docs[0].metadata["Section"] = "Contact Information"
#     for i in range(len(new_docs)):
#         if new_docs[i].metadata.get("Subsection") is None:
#             new_docs[i].metadata["Subsection"] = ""
#     return new_docs


# def vector_db(docs: list[Document] = [], drop_collection: bool = False):
#     uuids = [str(uuid4()) for _ in range(len(docs))]
#     vector_store.add_documents(documents=docs, ids=uuids)
#     # collection_name = "jason_rag_collection"
#     # partition_name = "resume_partition"

#     # # Drop the collection if requested
#     # if drop_collection and client.has_collection(collection_name=collection_name):
#     #     client.drop_collection(collection_name=collection_name)

#     # # Create the collection if it doesn't exist
#     # if not client.has_collection(collection_name=collection_name):
#     #     client.create_collection(collection_name=collection_name, dimension=768)

#     # # Encode the documents
#     # doc_vectors = embedding_model.encode_documents([doc.page_content for doc in docs])

#     # # Prepare the data for insertion
#     # data = [
#     #     {
#     #         "id": i,
#     #         "vector": doc_vectors[i].tolist(),  # convert numpy array to list
#     #         "text": docs[i].page_content,
#     #         "section": docs[i].metadata.get("Section", ""),
#     #         "subsection": docs[i].metadata.get("Subsection", ""),
#     #     }
#     #     for i in range(len(doc_vectors))
#     # ]

#     # res = client.insert(collection_name=collection_name, partition_name=partition_name, data=data)
#     # # Flush to ensure the data is persisted before indexing/searching
#     # client.flush(collection_name)

#     # print("Insert response:", res)


# async def query(q: str):
#     # within the query here you can probably employ a LLM to choose the specific section or subsection metadata filter by providing it those options and then the user query, to make retrieval more accurate
#     # res = vector_store.similarity_search_with_score(
#     #     q, k=3, ranker_type="rrf", ranker_params={"k": 100}
#     # )
#     res = vector_store._collection_hybrid_search(k=5, ranker_type="rrf", ranker_params={
#         "k": 100
#     }, query=q)
#     for hits in res:
#         for hit in hits:
#             print(hit.text)
#     # Encode the query
#     # embeddings = embedding_model.encode_queries([q])
#     # query_vector = embeddings[0].tolist()  # Convert to list
#     # # Note: Wrap the vector in a list (Milvus expects a list of vectors)
#     # res = client.search(
#     #     collection_name="jason_rag_collection",
#     #     data=[query_vector],
#     #     anns_field="vector",
#     #     limit=3,
#     #     output_fields=["text"],
#     # )
#     # Process and return the search results (adjust based on your schema)
#     # For example, extracting the 'text' field from each hit:
#     # results = []
#     # for result in res:
#     #     for hit in result:
#     #         results.append(hit["entity"]["text"])
#     # return results

# # def reranker(retriever):
# #     reranker_args = {
# #         'model': 'castorini/rank_zephyr_7b_v1_full',
# #         'top_n': 3,
# #     }
# #     reranker =
# #     compression_retriever = ContextualCompressionRetriever(
# #         base_compressor=compressor, base_retriever=retriever
# #     )

# #     return compression_retriever


# def main():
#     markdown_text = """
#     # Jason Wu\n\nwu80jason8@gmail.com | LinkedIn | GitHub | (925) 409-1051\n\n# EDUCATION\n\nSanta Clara University, Santa Clara, California\n\nB.S. in Computer Science and Engineering, Expected Graduation, June 2027\n\n- GPA: 3.90/4.0\n- Related Coursework: Data Structures & Algorithms, Physics for Engineers, Probability and Statistics for Engineers, Calculus I-IV, Object Oriented-Programming & Data Structures, Embedded Systems, Operating Systems\n\n# EXPERIENCE\n\n## Software Engineer Intern, Datatrixs — San Francisco, CA\n\nJuly 2024 – October 2024\n\n- Developed a full-stack SaaS application using React.js, Express.js, MongoDB, and AWS SDK to automate tasks for CPAs.\n- Programmed and maintained a serverless architecture by leveraging AWS services including S3, Lambda, and Cognito which reduced deployment and login times by 25%.\n- With OpenAI’s API, fine-tuned GPT-4o LLM to automate financial statement generation (Profit and Loss, Cash Flow, Balance Sheets, Income Statements), improving response accuracy by 50% and boosting client satisfaction by 20%.\n- Leveraged AWS S3 to integrate a file uploading feature, enabling users to upload custom financial data to the LLM.\n- Engineered an agentic RAG pipeline using OpenAI Assistant's API to automate chart and graph creation on CPA financial data, reducing creation time by 93% drastically cutting down on manual workload for clients.\n\n## Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA\n\nSeptember 2023 – July 2024\n\n- Under the supervision of Dr. Junho Park, implemented a digital twin environment utilizing machine learning, creating a bidirectional pipeline between the virtual and real world to provide more employment opportunities for amputees.\n- Developed and trained an eight-layer dense neural network and 1D CNN achieving 91% accuracy for multi-classification of muscle movements based on a dataset of EMG signals with TensorFlow.\n- Spearheaded the development of a RESTful API using FastAPI, enabling real-time transmission of EMG sensor data from Arduino to AR headset. Integrated a pipeline to display sensor data visually as muscle movements on AR headset display.\n- Deployed the API using Docker containers for availability and scalability, and hosted it on DigitalOcean to ensure service can support real-time data processing.\n\n## Software Engineer, AVBotz — Pleasanton, CA\n\nAugust 2021 – June 2023\n\n- Contributed to computer vision projects aimed at enhancing object detection capabilities for club's automated systems.\n- Designed a color detection system with Python and OpenCV to accurately identify red ellipses on a torpedo board. Applied solvePnP algorithm for precise 3D positioning from 2D images, achieving robust Euler angle determination.\n- Developed a HSV color filtering algorithm with OpenCV to enhance noise reduction underwater by 50% empowering Autonomous Underwater Vehicle (AUV) to precisely align with orange path markers based on angle and relative coordinates.\n- Achieved RoboSub 2022 Autonomy Challenge 2nd Place (International), while being the only high school team to participate in competition and beat out 37 other university teams (i.e. CMU, Duke, Cornell).\n\n# PROJECTS\n\n## Mind Over Matter\n\nJune 2024 – Present\n\n- Developed a full-stack RAG application aimed to support individuals who face mental health challenges, featuring personalized responses from an LLM based on uplifting quotes utilizing React.js, Tailwind, and Flask API.\n- Integrated a LLama3 model with LangChain to implement a RAG pipeline, using a ChromaDB vector store to store and retrieve quotes embeddings, enabling personalized and contextually relevant responses to user queries.\n- Implemented session-based user authentication, incorporating CSRF token authentication to prevent unauthorized actions.\n- Created an email verification pipeline with tokens and SendGrid API to authenticate registered users.\n\n## FastAPI To-Do List\n\nDecember 2023 – March 2024\n\n- Implemented a CRUD To-Do List application with Python, FastAPI, and SQLite. Features include creating and deleting todos, fetching an entire todo list from a database, checking off a todo, exporting as csv, and basic HTTP user authentication.\n\n# PUBLICATIONS\n\n- Wu, J., Jangid, V., & Park, J. Digital Twin for Amputees: A Bidirectional Interaction Modeling and Prototype with Convolutional Neural Network. Human Factors and Ergonomics Society, 2024, Link\n- Jangid, V., Sun, A., Wu, J., & Park, J. Ergonomic Augmented Reality Glasses Development for Workload Detection with Biofeedback Data and Machine Learning. Human Factors and Ergonomics Society, 2024, Link\n\n# SKILLS\n\nLanguages: Java, Python, C, C++, HTML/CSS, JavaScript, Bash, Verilog, SQL, TypeScript, Assembly\n\nTechnologies: Next.js, React.js, Node.js, Flask, Express.js, AWS, MongoDB, Docker, Machine Learning, Computer Vision, Firebase, LangChain, ChromaDB, ROS2, Git, Tensorflow, Linux, OpenCV, FastAPI, PostgreSQL, OAuth, Supabase, Web Sockets, Github Actions
#     """
#     new_res = chunk_documents(markdown_text)
#     # print(new_res)
#     vector_db(new_res, drop_collection=True)


# if __name__ == "__main__":
#     main()
#     asyncio.run(query("Where did Jason intern at last year?"))
