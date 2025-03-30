import os
from dotenv import load_dotenv

# from llama_cloud_services import LlamaParse
import asyncio
from langchain_core.documents import Document
import asyncpg
from langchain_community.document_loaders import TextLoader

# from langchain_qdrant.fastembed_sparse import FastEmbedSparse
# from langchain_huggingface import HuggingFaceEmbeddings
from asyncpg import Pool

# from langchain_community.vectorstores import SupabaseVectorStore
# from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter

# from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
from numpy import ndarray
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
import aiohttp

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

RESUME_DOCUMENT_CONTENT: str = """
# Jason Wu\n\nwu80jason8@gmail.com | LinkedIn (https://www.linkedin.com/in/jason-wu-261741215/) | GitHub (https://github.com/JDubWeuu) | (925) 409-1051\n\n# EDUCATION\n\nSanta Clara University, Santa Clara, California\n\nB.S. in Computer Science and Engineering, Expected Graduation, June 2027\n\n- GPA: 3.90/4.0\n- Related Coursework: Data Structures & Algorithms, Physics for Engineers, Probability and Statistics for Engineers, Calculus I-IV, Object Oriented-Programming & Data Structures, Embedded Systems, Operating Systems\n\n# EXPERIENCE\n\n## Software Engineer Intern, Datatrixs — San Francisco, CA\n\nJuly 2024 – October 2024\n\n- Developed a full-stack SaaS application using React.js, Express.js, MongoDB, and AWS SDK to automate tasks for CPAs.\n- Programmed and maintained a serverless architecture by leveraging AWS services including S3, Lambda, and Cognito which reduced deployment and login times by 25%.\n- With OpenAI's API, fine-tuned GPT-4o LLM to automate financial statement generation (Profit and Loss, Cash Flow, Balance Sheets, Income Statements), improving response accuracy by 50% and boosting client satisfaction by 20%.\n- Leveraged AWS S3 to integrate a file uploading feature, enabling users to upload custom financial data to the LLM.\n- Engineered an agentic RAG pipeline using OpenAI Assistant's API to automate chart and graph creation on CPA financial data, reducing creation time by 93% drastically cutting down on manual workload for clients.\n\n## Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA\n\nSeptember 2023 – July 2024\n\n- Under the supervision of Dr. Junho Park, implemented a digital twin environment utilizing machine learning, creating a bidirectional pipeline between the virtual and real world to provide more employment opportunities for amputees.\n- Developed and trained an eight-layer dense neural network and 1D CNN achieving 91% accuracy for multi-classification of muscle movements based on a dataset of EMG signals with TensorFlow.\n- Spearheaded the development of a RESTful API using FastAPI, enabling real-time transmission of EMG sensor data from Arduino to AR headset. Integrated a pipeline to display sensor data visually as muscle movements on AR headset display.\n- Deployed the API using Docker containers for availability and scalability, and hosted it on DigitalOcean to ensure service can support real-time data processing.\n\n## Software Engineer, AVBotz — Pleasanton, CA\n\nAugust 2021 – June 2023\n\n- Contributed to computer vision projects aimed at enhancing object detection capabilities for club's automated systems.\n- Designed a color detection system with Python and OpenCV to accurately identify red ellipses on a torpedo board. Applied solvePnP algorithm for precise 3D positioning from 2D images, achieving robust Euler angle determination.\n- Developed a HSV color filtering algorithm with OpenCV to enhance noise reduction underwater by 50% empowering Autonomous Underwater Vehicle (AUV) to precisely align with orange path markers based on angle and relative coordinates.\n- Achieved RoboSub 2022 Autonomy Challenge 2nd Place (International), while being the only high school team to participate in competition and beat out 37 other university teams (i.e. CMU, Duke, Cornell).\n\n# PROJECTS\n\n## Mind Over Matter\n\nJune 2024 – Present\n\n- Developed a full-stack RAG application aimed to support individuals who face mental health challenges, featuring personalized responses from an LLM based on uplifting quotes utilizing React.js, Tailwind, and Flask API.\n- Integrated a LLama3 model with LangChain to implement a RAG pipeline, using a ChromaDB vector store to store and retrieve quotes embeddings, enabling personalized and contextually relevant responses to user queries.\n- Implemented session-based user authentication, incorporating CSRF token authentication to prevent unauthorized actions.\n- Created an email verification pipeline with tokens and SendGrid API to authenticate registered users.\n\n## FastAPI To-Do List\n\nDecember 2023 – March 2024\n\n- Implemented a CRUD To-Do List application with Python, FastAPI, and SQLite. Features include creating and deleting todos, fetching an entire todo list from a database, checking off a todo, exporting as csv, and basic HTTP user authentication.\n\n# PUBLICATIONS\n\n- Wu, J., Jangid, V., & Park, J. Digital Twin for Amputees: A Bidirectional Interaction Modeling and Prototype with Convolutional Neural Network. Human Factors and Ergonomics Society, 2024, Link\n- Jangid, V., Sun, A., Wu, J., & Park, J. Ergonomic Augmented Reality Glasses Development for Workload Detection with Biofeedback Data and Machine Learning. Human Factors and Ergonomics Society, 2024, Link\n\n# SKILLS\n\nLanguages: Java, Python, C, C++, HTML/CSS, JavaScript, Bash, Verilog, SQL, TypeScript, Assembly\n\nTechnologies: Next.js, React.js, Node.js, Flask, Express.js, AWS, MongoDB, Docker, Machine Learning, Computer Vision, Firebase, LangChain, ChromaDB, ROS2, Git, Tensorflow, Linux, OpenCV, FastAPI, PostgreSQL, OAuth, Supabase, Web Sockets, Github Actions
"""


class PostgresRAG:
    def __init__(self) -> None:
        # self.parser = LlamaParse(
        #     result_type="markdown", api_key=os.getenv("LLAMA_CLOUD_API_KEY")
        # )
        self.url = os.getenv("SUPABASE_URL")
        self.db_url = os.getenv("SUPABASE_DB_URL")
        # self.supabase_client: Client = create_client(self.url, os.getenv("SUPABASE_API_KEY"))
        self.embedding_model_id = "Snowflake/snowflake-arctic-embed-l-v2.0"
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name=self.embedding_model_id,
        #     encode_kwargs={"normalize_embeddings": True},
        #     model_kwargs={"device": "cpu"},
        # )
        # self.embeddings_client = InferenceClient(
        #     model=self.embedding_model_id,
        #     api_key=os.getenv("HUGGING_FACE_API_KEY"),
        #     headers=
        # )
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.spec = ServerlessSpec(cloud="aws", region="us-east-1")
        self.index = self.pc.IndexAsyncio(
            host="https://hybridsearch-4r5zewn.svc.aped-4627-b74a.pinecone.io"
        )
        # self.clear_collection()
        # try:
        #     # ONLY UNCOMMENT THIS FOR TESTING, IN PRODUCTION MAKE SURE THE COLLECTION HAS ALL THE DOCUMENTS YOU NEED
        #     if self.qdrant_client.collection_exists("documents"):
        #         self.qdrant_client.delete_collection("documents")
        #     self.qdrant_client.create_collection("documents", vectors_config=VectorParams(
        #         size=768, distance=Distance.COSINE
        #     ))
        #     # print(self.qdrant_client.get_collections())
        # except Exception as e:
        #     print(f"Failed to create collection: {e}")
        # self.cohere_client = Cohere_Client(api_key=os.getenv("COHERE_API_KEY"))
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        # self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        # self.qdrant_client.recreate_collection("hybrid_docs", vectors_config=VectorParams())
        # self.vector_store: QdrantVectorStore = QdrantVectorStore(client=self.qdrant_client, collection_name="hybrid_docs", embedding=self.embeddings, sparse_embedding=self.sparse_embeddings, retrieval_mode=RetrievalMode.HYBRID)
        # self.vector_store: QdrantVectorStore = QdrantVectorStore.from_texts(texts=[], url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), collection_name="hybrid_docs", embedding=self.embeddings, sparse_embedding=self.sparse_embeddings, retrieval_mode=RetrievalMode.HYBRID, force_recreate=True)
        # print(self.vector_store.similarity_search(query="", k=12))
        # self.vector_store: SupabaseVectorStore = SupabaseVectorStore(client=self.supabase_client, table_name="documents", embedding=self.embeddings, query_name="match_documents")
        # self.docs: list[Document] = [Document(page_content=RESUME_DOCUMENT_CONTENT)]
        # self.docs = [Document(metadata={'Section': 'Contact Information', 'Subsection': ''}, page_content='# Jason Wu  \nwu80jason8@gmail.com | LinkedIn (https://www.linkedin.com/in/jason-wu-261741215/) | GitHub (https://github.com/JDubWeuu) | (925) 409-1051'), Document(metadata={'Section': 'EDUCATION', 'Subsection': ''}, page_content='# EDUCATION  \nSanta Clara University, Santa Clara, California  \nB.S. in Computer Science and Engineering, Expected Graduation, June 2027  \n- GPA: 3.90/4.0\n- Related Coursework: Data Structures & Algorithms, Physics for Engineers, Probability and Statistics for Engineers, Calculus I-IV, Object Oriented-Programming & Data Structures, Embedded Systems, Operating Systems'), Document(metadata={'Section': 'EXPERIENCE', 'Subsection': 'Software Engineer Intern, Datatrixs — San Francisco, CA'}, page_content="# EXPERIENCE  \n## Software Engineer Intern, Datatrixs — San Francisco, CA  \nJuly 2024 – October 2024  \n- Developed a full-stack SaaS application using React.js, Express.js, MongoDB, and AWS SDK to automate tasks for CPAs.\n- Programmed and maintained a serverless architecture by leveraging AWS services including S3, Lambda, and Cognito which reduced deployment and login times by 25%.\n- With OpenAI's API, fine-tuned GPT-4o LLM to automate financial statement generation (Profit and Loss, Cash Flow, Balance Sheets, Income Statements), improving response accuracy by 50% and boosting client satisfaction by 20%.\n- Leveraged AWS S3 to integrate a file uploading feature, enabling users to upload custom financial data to the LLM.\n- Engineered an agentic RAG pipeline using OpenAI Assistant's API to automate chart and graph creation on CPA financial data, reducing creation time by 93% drastically cutting down on manual workload for clients."), Document(metadata={'Section': 'EXPERIENCE', 'Subsection': 'Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA'}, page_content='## Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA  \nSeptember 2023 – July 2024  \n- Under the supervision of Dr. Junho Park, implemented a digital twin environment utilizing machine learning, creating a bidirectional pipeline between the virtual and real world to provide more employment opportunities for amputees.\n- Developed and trained an eight-layer dense neural network and 1D CNN achieving 91% accuracy for multi-classification of muscle movements based on a dataset of EMG signals with TensorFlow.\n- Spearheaded the development of a RESTful API using FastAPI, enabling real-time transmission of EMG sensor data from Arduino to AR headset. Integrated a pipeline to display sensor data visually as muscle movements on AR headset display.\n- Deployed the API using Docker containers for availability and scalability, and hosted it on DigitalOcean to ensure service can support real-time data processing.'), Document(metadata={'Section': 'EXPERIENCE', 'Subsection': 'Software Engineer, AVBotz — Pleasanton, CA'}, page_content="## Software Engineer, AVBotz — Pleasanton, CA  \nAugust 2021 – June 2023  \n- Contributed to computer vision projects aimed at enhancing object detection capabilities for club's automated systems.\n- Designed a color detection system with Python and OpenCV to accurately identify red ellipses on a torpedo board. Applied solvePnP algorithm for precise 3D positioning from 2D images, achieving robust Euler angle determination.\n- Developed a HSV color filtering algorithm with OpenCV to enhance noise reduction underwater by 50% empowering Autonomous Underwater Vehicle (AUV) to precisely align with orange path markers based on angle and relative coordinates.\n- Achieved RoboSub 2022 Autonomy Challenge 2nd Place (International), while being the only high school team to participate in competition and beat out 37 other university teams (i.e. CMU, Duke, Cornell)."), Document(metadata={'Section': 'PROJECTS', 'Subsection': 'Mind Over Matter'}, page_content='# PROJECTS  \n## Mind Over Matter  \nJune 2024 – Present  \n- Developed a full-stack RAG application aimed to support individuals who face mental health challenges, featuring personalized responses from an LLM based on uplifting quotes utilizing React.js, Tailwind, and Flask API.\n- Integrated a LLama3 model with LangChain to implement a RAG pipeline, using a ChromaDB vector store to store and retrieve quotes embeddings, enabling personalized and contextually relevant responses to user queries.\n- Implemented session-based user authentication, incorporating CSRF token authentication to prevent unauthorized actions.\n- Created an email verification pipeline with tokens and SendGrid API to authenticate registered users.'), Document(metadata={'Section': 'PROJECTS', 'Subsection': 'FastAPI To-Do List'}, page_content='## FastAPI To-Do List  \nDecember 2023 – March 2024  \n- Implemented a CRUD To-Do List application with Python, FastAPI, and SQLite. Features include creating and deleting todos, fetching an entire todo list from a database, checking off a todo, exporting as csv, and basic HTTP user authentication.'), Document(metadata={'Section': 'PUBLICATIONS', 'Subsection': ''}, page_content='# PUBLICATIONS  \n- Wu, J., Jangid, V., & Park, J. Digital Twin for Amputees: A Bidirectional Interaction Modeling and Prototype with Convolutional Neural Network. Human Factors and Ergonomics Society, 2024, Link\n- Jangid, V., Sun, A., Wu, J., & Park, J. Ergonomic Augmented Reality Glasses Development for Workload Detection with Biofeedback Data and Machine Learning. Human Factors and Ergonomics Society, 2024, Link'), Document(metadata={'Section': 'SKILLS', 'Subsection': ''}, page_content='# SKILLS  \nLanguages: Java, Python, C, C++, HTML/CSS, JavaScript, Bash, Verilog, SQL, TypeScript, Assembly  \nTechnologies: Next.js, React.js, Node.js, Flask, Express.js, AWS, MongoDB, Docker, Machine Learning, Computer Vision, Firebase, LangChain, ChromaDB, ROS2, Git, Tensorflow, Linux, OpenCV, FastAPI, PostgreSQL, OAuth, Supabase, Web Sockets, Github Actions'), Document(metadata={'Section': 'General Details', 'Subsection': ''}, page_content="#I'm Jason, a sophomore studying computer science and engineering at Santa Clara University.\nI'm passionate about being able to make a real-world impact on other people using the skills in software development I have.\nI'm always brainstorming and thinking about new cool projects I can take on, so if anyone has any, they can let me know.\nIn my free time, you can find me playing basketball with my friends or learning something new. However, if I'm not playing basketball, I could also be watching basketball.\nI serve on the board of the Cybersecurity Club on campus at Santa Clara University called BroncoSec. We try to provide knowledge of cybersecurity principles and development practices to those who are interested in the field."), Document(metadata={'Section': 'Nezerac', 'Subsection': ''}, page_content="# Nezerac\nOne of the projects that I developed was during the INRIX X AWS Hackathon in 2024 where myself and other teammates developed an app called Nezerac. Nezerac is an AI agent which helps ease the workload of restaurant owners.  It's use is to make it so closing deals on quality ingredients and other supplies, holding conversations via emails for the restaurant owner, so the owner can focus on serving the customers more than the supply side.\nI specifically worked on communication between the frontend and the backend, in writing and processing restaurant data via a lambda function into dynamodb for the AI Agent."), Document(metadata={'Section': 'Visionairy', 'Subsection': ''}, page_content='# Visionairy\nAnother one of the projects I developed called Visionairy was during Hack for Humanity 2025. In this hackathon, my team secured the Most Likely to Be a Startup prize at a hackathon with over 330+ participants from across California.\nI developed the backend using FastAPI, Langchain, Google Cloud Speech-to-Text Models, and Browser Use to automate and ease the booking process of flights for those who are visually impaired.')]
        self.pool: Pool = None
        self.bm25 = BM25Encoder.default()
        self.sparse_docs = None
        self.dense_docs = None

    async def dense_vector_embed(self, query: str) -> ndarray:
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv("HUGGING_FACE_API_KEY")}",
                "Content-Type": "application/json",
                "x-wait-for-model": "true",
                "x-use-cache": "true",
            }
            data = {"inputs": query}
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    "https://router.huggingface.co/hf-inference/pipeline/feature-extraction/Snowflake/snowflake-arctic-embed-l-v2.0",
                    headers=headers,
                    data=data,
                )
                dense_vectors = await response.json()
                return dense_vectors
        except Exception as e:
            raise e

    async def create_index(self):
        if "hybridsearch" in self.pc.list_indexes().names():
            self.pc.delete_index(name="hybridsearch")
        self.pc.create_index(
            name="hybridsearch",
            spec=self.spec,
            metric="dotproduct",  # for hybrid search, need to use dotproduct
            dimension=1024,
        )
        self.index = self.pc.Index("hybridsearch")
        self.index.describe_index_stats()

    async def initialize_vector_store(self) -> None:
        # self.clear_collection()
        # res = await self.vector_store.asearch(query="", search_type="similarity")
        # print(res[0])
        docs = [
            Document(
                metadata={"Section": "SKILLS"},
                page_content="# SKILLS  \nLanguages: Java, Python, C, C++, HTML/CSS, JavaScript, Bash, Verilog, SQL, TypeScript, Assembly  \nTechnologies: Next.js, React.js, Node.js, Flask, Express.js, AWS, MongoDB, Docker, Machine Learning, Computer Vision, Firebase, LangChain, ChromaDB, ROS2, Git, Tensorflow, Linux, OpenCV, FastAPI, PostgreSQL, OAuth, Supabase, Web Sockets, Github Actions",
            ),
            Document(
                metadata={
                    "Section": "EXPERIENCE",
                    "Subsection": "Software Engineer Intern, Datatrixs — San Francisco, CA",
                },
                page_content="# EXPERIENCE  \n## Software Engineer Intern, Datatrixs — San Francisco, CA  \nJuly 2024 – October 2024  \n- Developed a full-stack SaaS application using React.js, Express.js, MongoDB, and AWS SDK to automate tasks for CPAs.\n- Programmed and maintained a serverless architecture by leveraging AWS services including S3, Lambda, and Cognito which reduced deployment and login times by 25%.\n- With OpenAI's API, fine-tuned GPT-4o LLM to automate financial statement generation (Profit and Loss, Cash Flow, Balance Sheets, Income Statements), improving response accuracy by 50% and boosting client satisfaction by 20%.\n- Leveraged AWS S3 to integrate a file uploading feature, enabling users to upload custom financial data to the LLM.\n- Engineered an agentic RAG pipeline using OpenAI Assistant's API to automate chart and graph creation on CPA financial data, reducing creation time by 93% drastically cutting down on manual workload for clients.",
            ),
            Document(
                metadata={
                    "Section": "EXPERIENCE",
                    "Subsection": "Software Engineer, AVBotz — Pleasanton, CA",
                    "_id": "44cd9a8c-9700-479f-b6ed-c7bfbacbae79",
                    "_collection_name": "documents",
                },
                page_content="## Software Engineer, AVBotz — Pleasanton, CA  \nAugust 2021 – June 2023  \n- Contributed to computer vision projects aimed at enhancing object detection capabilities for club's automated systems.\n- Designed a color detection system with Python and OpenCV to accurately identify red ellipses on a torpedo board. Applied solvePnP algorithm for precise 3D positioning from 2D images, achieving robust Euler angle determination.\n- Developed a HSV color filtering algorithm with OpenCV to enhance noise reduction underwater by 50% empowering Autonomous Underwater Vehicle (AUV) to precisely align with orange path markers based on angle and relative coordinates.\n- Achieved RoboSub 2022 Autonomy Challenge 2nd Place (International), while being the only high school team to participate in competition and beat out 37 other university teams (i.e. CMU, Duke, Cornell).",
            ),
            Document(
                metadata={
                    "Section": "PROJECTS",
                    "Subsection": "FastAPI To-Do List",
                    "_id": "e769d1f3-0afa-4122-ac72-f49821b9cf6c",
                    "_collection_name": "documents",
                },
                page_content="## FastAPI To-Do List  \nDecember 2023 – March 2024  \n- Implemented a CRUD To-Do List application with Python, FastAPI, and SQLite. Features include creating and deleting todos, fetching an entire todo list from a database, checking off a todo, exporting as csv, and basic HTTP user authentication.",
            ),
            Document(
                metadata={
                    "Section": "EXPERIENCE",
                    "Subsection": "Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA",
                    "_id": "59f5c3c0-30f0-424a-b22b-45853cdb1284",
                    "_collection_name": "documents",
                },
                page_content="## Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA  \nSeptember 2023 – July 2024  \n- Under the supervision of Dr. Junho Park, implemented a digital twin environment utilizing machine learning, creating a bidirectional pipeline between the virtual and real world to provide more employment opportunities for amputees.\n- Developed and trained an eight-layer dense neural network and 1D CNN achieving 91% accuracy for multi-classification of muscle movements based on a dataset of EMG signals with TensorFlow.\n- Spearheaded the development of a RESTful API using FastAPI, enabling real-time transmission of EMG sensor data from Arduino to AR headset. Integrated a pipeline to display sensor data visually as muscle movements on AR headset display.\n- Deployed the API using Docker containers for availability and scalability, and hosted it on DigitalOcean to ensure service can support real-time data processing.",
            ),
            Document(
                metadata={
                    "Section": "PUBLICATIONS",
                    "Subsection": "",
                    "_id": "1dbce2da-5da0-4fe5-97a2-55e6200371d0",
                    "_collection_name": "documents",
                },
                page_content="# PUBLICATIONS  \n- Wu, J., Jangid, V., & Park, J. Digital Twin for Amputees: A Bidirectional Interaction Modeling and Prototype with Convolutional Neural Network. Human Factors and Ergonomics Society, 2024, Link\n- Jangid, V., Sun, A., Wu, J., & Park, J. Ergonomic Augmented Reality Glasses Development for Workload Detection with Biofeedback Data and Machine Learning. Human Factors and Ergonomics Society, 2024, Link",
            ),
            Document(
                metadata={
                    "Section": "PROJECTS",
                    "Subsection": "Mind Over Matter",
                    "_id": "bc088843-7baf-467b-a77c-1e308e3fe4ae",
                    "_collection_name": "documents",
                },
                page_content="# PROJECTS  \n## Mind Over Matter  \nJune 2024 – Present  \n- Developed a full-stack RAG application aimed to support individuals who face mental health challenges, featuring personalized responses from an LLM based on uplifting quotes utilizing React.js, Tailwind, and Flask API.\n- Integrated a LLama3 model with LangChain to implement a RAG pipeline, using a ChromaDB vector store to store and retrieve quotes embeddings, enabling personalized and contextually relevant responses to user queries.\n- Implemented session-based user authentication, incorporating CSRF token authentication to prevent unauthorized actions.\n- Created an email verification pipeline with tokens and SendGrid API to authenticate registered users.",
            ),
            Document(
                metadata={
                    "Section": "Visionairy",
                    "Subsection": "",
                    "_id": "fac39cae-57de-4eff-8375-20e18b0ffb64",
                    "_collection_name": "documents",
                },
                page_content="# Visionairy\nAnother one of the projects I developed called Visionairy was during Hack for Humanity 2025. In this hackathon, my team secured the Most Likely to Be a Startup prize at a hackathon with over 330+ participants from across California.\nI developed the backend using FastAPI, Langchain, Google Cloud Speech-to-Text Models, and Browser Use to automate and ease the booking process of flights for those who are visually impaired.",
            ),
            Document(
                metadata={
                    "Section": "Contact Information",
                    "Subsection": "",
                    "_id": "c3885f83-25ca-4517-a54f-71449a4ca053",
                    "_collection_name": "documents",
                },
                page_content="# Jason Wu  \nwu80jason8@gmail.com | LinkedIn (https://www.linkedin.com/in/jason-wu-261741215/) | GitHub (https://github.com/JDubWeuu) | (925) 409-1051",
            ),
            Document(
                metadata={
                    "Section": "General Details",
                    "Subsection": "",
                    "_id": "0a7f603b-9512-435b-9a91-8d1bf3278c73",
                    "_collection_name": "documents",
                },
                page_content="I'm Jason, a sophomore studying computer science and engineering at Santa Clara University.\nI'm passionate about being able to make a real-world impact on other people using the skills in software development I have.\nIn my free time, you can find me playing basketball with my friends or learning something new. However, if I'm not playing basketball, I could also be watching basketball.\nI serve on the board of the Cybersecurity Club on campus at Santa Clara University called BroncoSec. We try to provide knowledge of cybersecurity principles and development practices to those who are interested in the field.",
            ),
            Document(
                metadata={
                    "Section": "EDUCATION",
                    "Subsection": "",
                    "_id": "47db2887-7c34-4c10-ba9d-71aa868fa477",
                    "_collection_name": "documents",
                },
                page_content="# EDUCATION  \nSanta Clara University, Santa Clara, California  \nB.S. in Computer Science and Engineering, Expected Graduation, June 2027  \n- GPA: 3.90/4.0\n- Related Coursework: Data Structures & Algorithms, Physics for Engineers, Probability and Statistics for Engineers, Calculus I-IV, Object Oriented-Programming & Data Structures, Embedded Systems, Operating Systems",
            ),
            Document(
                metadata={
                    "Section": "Nezerac",
                    "Subsection": "",
                    "_collection_name": "documents",
                },
                page_content="# Nezerac\nOne of the projects that I developed was during the INRIX X AWS Hackathon in 2024 where myself and other teammates developed an app called Nezerac. Nezerac is an AI agent which helps ease the workload of restaurant owners.  It's use is to make it so closing deals on quality ingredients and other supplies, holding conversations via emails for the restaurant owner, so the owner can focus on serving the customers more than the supply side.\nI specifically worked on communication between the frontend and the backend, in writing and processing restaurant data via a lambda function into dynamodb for the AI Agent.",
            ),
        ]
        new_docs = []
        for doc in docs:
            doc.metadata.pop("_id", None)
            doc.metadata.pop("_collection_name", None)
            new_docs.append(doc)
        self.docs = new_docs
        self.create_dense_vectors(new_docs)
        self.create_sparse_vectors(new_docs)
        # print(new_docs)
        # res = await self.vector_store.aadd_documents(documents=new_docs)
        # print(res)

    def create_sparse_vectors(self, docs: list[Document]):
        res = self.bm25.encode_documents(texts=[doc.page_content for doc in docs])
        self.sparse_docs = res

    def create_dense_vectors(self, docs: list[Document]):
        res = [self.dense_vector_embed(doc) for doc in docs]
        # res = await self.embeddings.aembed_documents(docs)
        self.dense_docs = res

    async def embed_query(self, query: str):
        return self.bm25.encode_queries(query)

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.db_url, statement_cache_size=0)

    # async def parse_pdf(self, file_name: str) -> bool:
    #     try:
    #         BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #         file_path = os.path.join(
    #             BASE_DIR, "../assets/Jason_Wu_Resume.pdf"
    #         )  # Resolve absolute path
    #         reader = SimpleDirectoryReader(
    #             input_files=[file_path], file_extractor={".pdf": self.parser}
    #         )
    #         docs = await reader.aload_data()
    #         self.docs.extend(docs)
    #         return True
    #     except Exception as e:
    #         raise e

    async def parse_markdown(self, file_name: str):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(
            BASE_DIR, f"../assets/{file_name}"
        )  # Resolve absolute path
        doc_loader = TextLoader(file_path)
        docs = await doc_loader.aload()
        self.docs.extend(docs)
        # await self.chunk_docs(use_c_splitter=True, strip_headers=True)
        # print(docs)

    async def clear_table(self) -> None:
        await self.connect()
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE documents;")

    async def chunk_docs(
        self, use_c_splitter: bool = False, strip_headers: bool = False
    ) -> None:
        # First split by headers to maintain document structure
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Section"), ("##", "Subsection")],
            strip_headers=strip_headers,
        )

        new_docs: list[Document] = []

        # Process each document
        for doc in self.docs:
            content = doc.page_content or doc.text
            # First split by headers
            header_splits = header_splitter.split_text(content)

            # Then further split large chunks
            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=500, chunk_overlap=100, length_function=len
            )

            for header_doc in header_splits:
                # Preserve header metadata
                section = header_doc.metadata.get("Section", "")
                subsection = header_doc.metadata.get("Subsection", "")

                # Split large chunks while preserving metadata
                if (
                    len(header_doc.page_content) > 600
                ):  # Only split if chunk is too large
                    smaller_chunks = text_splitter.split_text(header_doc.page_content)
                    for chunk in smaller_chunks:
                        new_docs.append(
                            Document(
                                page_content=chunk,
                                metadata={"Section": section, "Subsection": subsection},
                            )
                        )
                else:
                    new_docs.append(header_doc)

        # Clean up General Details section
        for i in range(len(new_docs)):
            if "# General Details\\n" in new_docs[i].page_content:
                new_docs[i].page_content = new_docs[i].page_content.split(
                    "# General Details\\n"
                )[1]
        print("The length of the new docs is", len(new_docs))
        if new_docs:
            new_docs[0].metadata["Section"] = "Contact Information"
        for doc in new_docs:
            if doc.metadata.get("Subsection") is None:
                doc.metadata["Subsection"] = ""

        self.docs = new_docs
        print(f"Created {len(new_docs)} chunks")

    def upsert_docs_into_pc(self) -> None:
        print("UPSERTING DOCS")
        print("---------------------------")
        ids = [str(i) for i in range(len(self.docs))]

        upserts = []
        for _id, sparse_vec, dense_vec, doc in zip(
            ids, self.sparse_docs, self.dense_docs, self.docs
        ):
            upserts.append(
                {
                    "id": _id,
                    "sparse_values": sparse_vec,
                    "values": dense_vec,
                    "metadata": {
                        "section": doc.metadata.get("Section", ""),
                        "subsection": doc.metadata.get("Subsection", ""),
                        "content": doc.page_content,
                    },
                }
            )

        self.index.upsert(upserts)
        print(self.index.describe_index_stats())

    def hybrid_scale(self, dense, sparse, alpha: float):
        """Hybrid vector scaling using a convex combination

        alpha * dense + (1 - alpha) * sparse

        Args:
            dense: Array of floats representing
            sparse: a dict of `indices` and `values`
            alpha: float between 0 and 1 where 0 == sparse only
                and 1 == dense only
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        # scale sparse and dense vectors to create hybrid search vecs
        hsparse = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse

    async def hybrid_search(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> list[dict]:

        sparse_query = self.bm25.encode_queries(texts=query)
        dense_query = await self.dense_vector_embed(query=query)

        hdense, hsparse = self.hybrid_scale(dense_query, sparse_query, alpha)

        print("embedded queries")

        # Perform hybrid search
        results = await self.index.query(
            vector=hdense,
            sparse_vector=hsparse,
            top_k=top_k,
            include_metadata=True,
        )

        # Rerank with more context
        # ranked_docs = self.pc.inference.rerank(
        #     model="bge-reranker-v2-m3",
        #     documents=[doc['metadata']['content'] for doc in results['matches']],
        #     query=query,
        #     top_n=top_k,
        #     return_documents=True
        # )

        # print("hybrid search finished")

        return results["matches"]

    # async def close_connection(self):
    #     if self.pool:
    #         await self.pool.close()
    #     if self.qdrant_client:
    #         self.qdrant_client.close()

    async def close(self):
        await self.index.close()


async def main() -> None:
    db = PostgresRAG()
    matches = await db.hybrid_search("What internships has Jason done?", alpha=0.6)
    print(matches)
    await db.close()
    # await db.create_index()
    # db.initialize_vector_store()
    # db.upsert_docs_into_pc()
    # print(db.dense_docs)
    # print("-----------")
    # print(db.sparse_docs)


if __name__ == "__main__":
    # db = PostgresRAG()
    asyncio.run(main())
