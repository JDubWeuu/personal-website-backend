import os
from dotenv import load_dotenv
from supabase import create_client, Client
from llama_cloud_services import LlamaParse
import asyncio
from llama_index.core import SimpleDirectoryReader
import numpy as np
from langchain_core.documents import Document
import asyncpg
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import JinaEmbeddings

load_dotenv()

class PostgresRAG:
    def __init__(self) -> None:
        self.parser = LlamaParse(
            result_type="markdown",
            api_key=os.getenv("LLAMA_CLOUD_API_KEY")
        )
        self.url = os.getenv("SUPABASE_URL")
        self.db_url = os.getenv("SUPABASE_DB_URL")
        self.supabase_client: Client = create_client(self.url, os.getenv("SUPABASE_API_KEY"))
        self.embeddings = JinaEmbeddings(api_key=os.getenv("JINA_API_KEY"))
        self.vector_store: SupabaseVectorStore = SupabaseVectorStore(client=self.supabase_client, table_name="documents", embedding=self.embeddings, query_name="match_documents")
        self.docs: list[Document] = [Document(page_content="# Jason Wu\n\nwu80jason8@gmail.com | LinkedIn | GitHub | (925) 409-1051\n\n# EDUCATION\n\nSanta Clara University, Santa Clara, California\n\nB.S. in Computer Science and Engineering, Expected Graduation, June 2027\n\n- GPA: 3.90/4.0\n- Related Coursework: Data Structures & Algorithms, Physics for Engineers, Probability and Statistics for Engineers, Calculus I-IV, Object Oriented-Programming & Data Structures, Embedded Systems, Operating Systems\n\n# EXPERIENCE\n\n# Software Engineer Intern, Datatrixs — San Francisco, CA\n\nJuly 2024 – October 2024\n\n- Developed a full-stack SaaS application using React.js, Express.js, MongoDB, and AWS SDK to automate tasks for CPAs.\n- Programmed and maintained a serverless architecture by leveraging AWS services including S3, Lambda, and Cognito which reduced deployment and login times by 25%.\n- With OpenAI’s API, fine-tuned GPT-4o LLM to automate financial statement generation (Profit and Loss, Cash Flow, Balance Sheets, Income Statements), improving response accuracy by 50% and boosting client satisfaction by 20%.\n- Leveraged AWS S3 to integrate a file uploading feature, enabling users to upload custom financial data to the LLM.\n- Engineered an agentic RAG pipeline using OpenAI Assistant's API to automate chart and graph creation on CPA financial data, reducing creation time by 93% drastically cutting down on manual workload for clients.\n\n# Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA\n\nSeptember 2023 – July 2024\n\n- Under the supervision of Dr. Junho Park, implemented a digital twin environment utilizing machine learning, creating a bidirectional pipeline between the virtual and real world to provide more employment opportunities for amputees.\n- Developed and trained an eight-layer dense neural network and 1D CNN achieving 91% accuracy for multi-classification of muscle movements based on a dataset of EMG signals with TensorFlow.\n- Spearheaded the development of a RESTful API using FastAPI, enabling real-time transmission of EMG sensor data from Arduino to AR headset. Integrated a pipeline to display sensor data visually as muscle movements on AR headset display.\n- Deployed the API using Docker containers for availability and scalability, and hosted it on DigitalOcean to ensure service can support real-time data processing.\n\n# Software Engineer, AVBotz — Pleasanton, CA\n\nAugust 2021 – June 2023\n\n- Contributed to computer vision projects aimed at enhancing object detection capabilities for club's automated systems.\n- Designed a color detection system with Python and OpenCV to accurately identify red ellipses on a torpedo board. Applied solvePnP algorithm for precise 3D positioning from 2D images, achieving robust Euler angle determination.\n- Developed a HSV color filtering algorithm with OpenCV to enhance noise reduction underwater by 50% empowering Autonomous Underwater Vehicle (AUV) to precisely align with orange path markers based on angle and relative coordinates.\n- Achieved RoboSub 2022 Autonomy Challenge 2nd Place (International), while being the only high school team to participate in competition and beat out 37 other university teams (i.e. CMU, Duke, Cornell).\n\n# PROJECTS\n\n# Mind Over Matter\n\nJune 2024 – Present\n\n- Developed a full-stack RAG application aimed to support individuals who face mental health challenges, featuring personalized responses from an LLM based on uplifting quotes utilizing React.js, Tailwind, and Flask API.\n- Integrated a LLama3 model with LangChain to implement a RAG pipeline, using a ChromaDB vector store to store and retrieve quotes embeddings, enabling personalized and contextually relevant responses to user queries.\n- Implemented session-based user authentication, incorporating CSRF token authentication to prevent unauthorized actions.\n- Created an email verification pipeline with tokens and SendGrid API to authenticate registered users.\n\n# FastAPI To-Do List\n\nDecember 2023 – March 2024\n\n- Implemented a CRUD To-Do List application with Python, FastAPI, and SQLite. Features include creating and deleting todos, fetching an entire todo list from a database, checking off a todo, exporting as csv, and basic HTTP user authentication.\n\n# PUBLICATIONS\n\n- Wu, J., Jangid, V., & Park, J. Digital Twin for Amputees: A Bidirectional Interaction Modeling and Prototype with Convolutional Neural Network. Human Factors and Ergonomics Society, 2024, Link\n- Jangid, V., Sun, A., Wu, J., & Park, J. Ergonomic Augmented Reality Glasses Development for Workload Detection with Biofeedback Data and Machine Learning. Human Factors and Ergonomics Society, 2024, Link\n\n# SKILLS\n\nLanguages: Java, Python, C, C++, HTML/CSS, JavaScript, Bash, Verilog, SQL, TypeScript, Assembly\n\nTechnologies: Next.js, React.js, Node.js, Flask, Express.js, AWS, MongoDB, Docker, Machine Learning, Computer Vision, Firebase, LangChain, ChromaDB, ROS2, Git, Tensorflow, Linux, OpenCV, FastAPI, PostgreSQL, OAuth, Supabase, Web Sockets, Github Actions")]
        self.pool = None
        
    async def initialize_vector_store(self) -> None:
        res = await self.vector_store.aadd_documents(self.docs)
        print(res)
    
    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.db_url, statement_cache_size=0)
    
    async def query(self, q: str, top_k: int = 3) -> list[Document]:
        query_embedding = await self.embeddings.aembed_query(q)
        query_embedding = "[" + ",".join(map(str, query_embedding)) + "]"
        async with self.pool.acquire() as conn:
            async with conn.transaction():  # Ensures a safe transaction block
                rows = await conn.fetch(
                        """
                        SELECT id, content, metadata, similarity::double precision, 
                            text_rank::double precision, hybrid_score::double precision
                        FROM match_documents($1::vector, $2::TEXT, '{}')
                        ORDER BY hybrid_score DESC
                        LIMIT $3;
                        """,
                        query_embedding,  # Ensure this is 1536-dim
                        q,                # Full-text search query
                        top_k             
                )
                return [
                    Document(page_content=row["content"], metadata=dict(row["metadata"])) for row in rows
                ]
    async def parse_pdf(self, file_names: list[str]) -> bool:
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(BASE_DIR, "../assets/Jason_Wu_Resume.pdf")  # Resolve absolute path
            reader = SimpleDirectoryReader(input_files=[file_path], file_extractor={".pdf": self.parser})
            docs = await reader.aload_data()
            self.docs.extend(docs)
            return True
        except Exception as e:
            raise e
    
    def clear_vector_store(self) -> None:
        self.vector_store = None
    
    async def chunk_docs(self) -> None:
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Section"), ("##", "Subsection")], strip_headers=False)
        new_docs = None
        for doc in self.docs:
            content = doc.page_content or doc.text
            new_docs = splitter.split_text(content)
        print("The length of the new docs is", len(new_docs))
        if new_docs:
            new_docs[0].metadata["Section"] = "Contact Information"
        for i in range(len(new_docs)):
            if new_docs[i].metadata.get("Subsection") is None:
                new_docs[i].metadata["Subsection"] = ""
        self.docs = new_docs
    
    async def close_connection(self):
        if self.pool:
            await self.pool.close()

async def main() -> None:
    db = PostgresRAG()
    await db.connect()
    
    await db.chunk_docs()
    await db.initialize_vector_store()
    results = await db.query("What courses has Jason taken so far?")
    
    print(results)
    
    await db.close_connection()
    
if __name__ == "__main__":
    # db = PostgresRAG()
    # # asyncio.run(db.parse_pdf("/Users/jasonwu/personal_website_backend/app/assets/Jason_Wu_Resume.pdf"))
    # asyncio.run(db.chunk_docs())
    # asyncio.run(db.connect())
    # asyncio.run(db.query("What courses has Jason taken so far?"))
    asyncio.run(main())
    
    