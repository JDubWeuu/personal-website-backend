import os
from dotenv import load_dotenv
from supabase import create_client, Client
from llama_cloud_services import LlamaParse
import asyncio
from llama_index.core import SimpleDirectoryReader
from langchain_core.documents import Document
import asyncpg
from thefuzz import fuzz
from qdrant_client import QdrantClient
import numpy as np
from nltk.corpus import stopwords
from cohere import AsyncClientV2 as Cohere_Client
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from nltk.tokenize import word_tokenize
from asyncpg import Pool
# from langchain_community.vectorstores import SupabaseVectorStore
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings

load_dotenv()

RESUME_DOCUMENT_CONTENT: str = """
# Jason Wu\n\nwu80jason8@gmail.com | LinkedIn (https://www.linkedin.com/in/jason-wu-261741215/) | GitHub (https://github.com/JDubWeuu) | (925) 409-1051\n\n# EDUCATION\n\nSanta Clara University, Santa Clara, California\n\nB.S. in Computer Science and Engineering, Expected Graduation, June 2027\n\n- GPA: 3.90/4.0\n- Related Coursework: Data Structures & Algorithms, Physics for Engineers, Probability and Statistics for Engineers, Calculus I-IV, Object Oriented-Programming & Data Structures, Embedded Systems, Operating Systems\n\n# EXPERIENCE\n\n## Software Engineer Intern, Datatrixs — San Francisco, CA\n\nJuly 2024 – October 2024\n\n- Developed a full-stack SaaS application using React.js, Express.js, MongoDB, and AWS SDK to automate tasks for CPAs.\n- Programmed and maintained a serverless architecture by leveraging AWS services including S3, Lambda, and Cognito which reduced deployment and login times by 25%.\n- With OpenAI’s API, fine-tuned GPT-4o LLM to automate financial statement generation (Profit and Loss, Cash Flow, Balance Sheets, Income Statements), improving response accuracy by 50% and boosting client satisfaction by 20%.\n- Leveraged AWS S3 to integrate a file uploading feature, enabling users to upload custom financial data to the LLM.\n- Engineered an agentic RAG pipeline using OpenAI Assistant's API to automate chart and graph creation on CPA financial data, reducing creation time by 93% drastically cutting down on manual workload for clients.\n\n## Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA\n\nSeptember 2023 – July 2024\n\n- Under the supervision of Dr. Junho Park, implemented a digital twin environment utilizing machine learning, creating a bidirectional pipeline between the virtual and real world to provide more employment opportunities for amputees.\n- Developed and trained an eight-layer dense neural network and 1D CNN achieving 91% accuracy for multi-classification of muscle movements based on a dataset of EMG signals with TensorFlow.\n- Spearheaded the development of a RESTful API using FastAPI, enabling real-time transmission of EMG sensor data from Arduino to AR headset. Integrated a pipeline to display sensor data visually as muscle movements on AR headset display.\n- Deployed the API using Docker containers for availability and scalability, and hosted it on DigitalOcean to ensure service can support real-time data processing.\n\n## Software Engineer, AVBotz — Pleasanton, CA\n\nAugust 2021 – June 2023\n\n- Contributed to computer vision projects aimed at enhancing object detection capabilities for club's automated systems.\n- Designed a color detection system with Python and OpenCV to accurately identify red ellipses on a torpedo board. Applied solvePnP algorithm for precise 3D positioning from 2D images, achieving robust Euler angle determination.\n- Developed a HSV color filtering algorithm with OpenCV to enhance noise reduction underwater by 50% empowering Autonomous Underwater Vehicle (AUV) to precisely align with orange path markers based on angle and relative coordinates.\n- Achieved RoboSub 2022 Autonomy Challenge 2nd Place (International), while being the only high school team to participate in competition and beat out 37 other university teams (i.e. CMU, Duke, Cornell).\n\n# PROJECTS\n\n## Mind Over Matter\n\nJune 2024 – Present\n\n- Developed a full-stack RAG application aimed to support individuals who face mental health challenges, featuring personalized responses from an LLM based on uplifting quotes utilizing React.js, Tailwind, and Flask API.\n- Integrated a LLama3 model with LangChain to implement a RAG pipeline, using a ChromaDB vector store to store and retrieve quotes embeddings, enabling personalized and contextually relevant responses to user queries.\n- Implemented session-based user authentication, incorporating CSRF token authentication to prevent unauthorized actions.\n- Created an email verification pipeline with tokens and SendGrid API to authenticate registered users.\n\n## FastAPI To-Do List\n\nDecember 2023 – March 2024\n\n- Implemented a CRUD To-Do List application with Python, FastAPI, and SQLite. Features include creating and deleting todos, fetching an entire todo list from a database, checking off a todo, exporting as csv, and basic HTTP user authentication.\n\n# PUBLICATIONS\n\n- Wu, J., Jangid, V., & Park, J. Digital Twin for Amputees: A Bidirectional Interaction Modeling and Prototype with Convolutional Neural Network. Human Factors and Ergonomics Society, 2024, Link\n- Jangid, V., Sun, A., Wu, J., & Park, J. Ergonomic Augmented Reality Glasses Development for Workload Detection with Biofeedback Data and Machine Learning. Human Factors and Ergonomics Society, 2024, Link\n\n# SKILLS\n\nLanguages: Java, Python, C, C++, HTML/CSS, JavaScript, Bash, Verilog, SQL, TypeScript, Assembly\n\nTechnologies: Next.js, React.js, Node.js, Flask, Express.js, AWS, MongoDB, Docker, Machine Learning, Computer Vision, Firebase, LangChain, ChromaDB, ROS2, Git, Tensorflow, Linux, OpenCV, FastAPI, PostgreSQL, OAuth, Supabase, Web Sockets, Github Actions
"""

class PostgresRAG:
    def __init__(self) -> None:
        self.parser = LlamaParse(
            result_type="markdown",
            api_key=os.getenv("LLAMA_CLOUD_API_KEY")
        )
        self.url = os.getenv("SUPABASE_URL")
        self.db_url = os.getenv("SUPABASE_DB_URL")
        # self.supabase_client: Client = create_client(self.url, os.getenv("SUPABASE_API_KEY"))
        self.qdrant_client: QdrantClient = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
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
        self.cohere_client = Cohere_Client(api_key=os.getenv("COHERE_API_KEY"))
        self.embeddings = JinaEmbeddings(api_key=os.getenv("JINA_API_KEY"))
        self.vector_store: QdrantVectorStore = QdrantVectorStore(client=self.qdrant_client, collection_name="documents", embedding=self.embeddings)
        # self.vector_store: SupabaseVectorStore = SupabaseVectorStore(client=self.supabase_client, table_name="documents", embedding=self.embeddings, query_name="match_documents")
        # self.docs: list[Document] = [Document(page_content=RESUME_DOCUMENT_CONTENT)]
        self.docs = [Document(metadata={'Section': 'Contact Information', 'Subsection': ''}, page_content='# Jason Wu  \nwu80jason8@gmail.com | LinkedIn (https://www.linkedin.com/in/jason-wu-261741215/) | GitHub (https://github.com/JDubWeuu) | (925) 409-1051'), Document(metadata={'Section': 'EDUCATION', 'Subsection': ''}, page_content='# EDUCATION  \nSanta Clara University, Santa Clara, California  \nB.S. in Computer Science and Engineering, Expected Graduation, June 2027  \n- GPA: 3.90/4.0\n- Related Coursework: Data Structures & Algorithms, Physics for Engineers, Probability and Statistics for Engineers, Calculus I-IV, Object Oriented-Programming & Data Structures, Embedded Systems, Operating Systems'), Document(metadata={'Section': 'EXPERIENCE', 'Subsection': 'Software Engineer Intern, Datatrixs — San Francisco, CA'}, page_content="# EXPERIENCE  \n## Software Engineer Intern, Datatrixs — San Francisco, CA  \nJuly 2024 – October 2024  \n- Developed a full-stack SaaS application using React.js, Express.js, MongoDB, and AWS SDK to automate tasks for CPAs.\n- Programmed and maintained a serverless architecture by leveraging AWS services including S3, Lambda, and Cognito which reduced deployment and login times by 25%.\n- With OpenAI’s API, fine-tuned GPT-4o LLM to automate financial statement generation (Profit and Loss, Cash Flow, Balance Sheets, Income Statements), improving response accuracy by 50% and boosting client satisfaction by 20%.\n- Leveraged AWS S3 to integrate a file uploading feature, enabling users to upload custom financial data to the LLM.\n- Engineered an agentic RAG pipeline using OpenAI Assistant's API to automate chart and graph creation on CPA financial data, reducing creation time by 93% drastically cutting down on manual workload for clients."), Document(metadata={'Section': 'EXPERIENCE', 'Subsection': 'Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA'}, page_content='## Undergraduate Researcher, Human-AI Systems Optimization Lab — Santa Clara, CA  \nSeptember 2023 – July 2024  \n- Under the supervision of Dr. Junho Park, implemented a digital twin environment utilizing machine learning, creating a bidirectional pipeline between the virtual and real world to provide more employment opportunities for amputees.\n- Developed and trained an eight-layer dense neural network and 1D CNN achieving 91% accuracy for multi-classification of muscle movements based on a dataset of EMG signals with TensorFlow.\n- Spearheaded the development of a RESTful API using FastAPI, enabling real-time transmission of EMG sensor data from Arduino to AR headset. Integrated a pipeline to display sensor data visually as muscle movements on AR headset display.\n- Deployed the API using Docker containers for availability and scalability, and hosted it on DigitalOcean to ensure service can support real-time data processing.'), Document(metadata={'Section': 'EXPERIENCE', 'Subsection': 'Software Engineer, AVBotz — Pleasanton, CA'}, page_content="## Software Engineer, AVBotz — Pleasanton, CA  \nAugust 2021 – June 2023  \n- Contributed to computer vision projects aimed at enhancing object detection capabilities for club's automated systems.\n- Designed a color detection system with Python and OpenCV to accurately identify red ellipses on a torpedo board. Applied solvePnP algorithm for precise 3D positioning from 2D images, achieving robust Euler angle determination.\n- Developed a HSV color filtering algorithm with OpenCV to enhance noise reduction underwater by 50% empowering Autonomous Underwater Vehicle (AUV) to precisely align with orange path markers based on angle and relative coordinates.\n- Achieved RoboSub 2022 Autonomy Challenge 2nd Place (International), while being the only high school team to participate in competition and beat out 37 other university teams (i.e. CMU, Duke, Cornell)."), Document(metadata={'Section': 'PROJECTS', 'Subsection': 'Mind Over Matter'}, page_content='# PROJECTS  \n## Mind Over Matter  \nJune 2024 – Present  \n- Developed a full-stack RAG application aimed to support individuals who face mental health challenges, featuring personalized responses from an LLM based on uplifting quotes utilizing React.js, Tailwind, and Flask API.\n- Integrated a LLama3 model with LangChain to implement a RAG pipeline, using a ChromaDB vector store to store and retrieve quotes embeddings, enabling personalized and contextually relevant responses to user queries.\n- Implemented session-based user authentication, incorporating CSRF token authentication to prevent unauthorized actions.\n- Created an email verification pipeline with tokens and SendGrid API to authenticate registered users.'), Document(metadata={'Section': 'PROJECTS', 'Subsection': 'FastAPI To-Do List'}, page_content='## FastAPI To-Do List  \nDecember 2023 – March 2024  \n- Implemented a CRUD To-Do List application with Python, FastAPI, and SQLite. Features include creating and deleting todos, fetching an entire todo list from a database, checking off a todo, exporting as csv, and basic HTTP user authentication.'), Document(metadata={'Section': 'PUBLICATIONS', 'Subsection': ''}, page_content='# PUBLICATIONS  \n- Wu, J., Jangid, V., & Park, J. Digital Twin for Amputees: A Bidirectional Interaction Modeling and Prototype with Convolutional Neural Network. Human Factors and Ergonomics Society, 2024, Link\n- Jangid, V., Sun, A., Wu, J., & Park, J. Ergonomic Augmented Reality Glasses Development for Workload Detection with Biofeedback Data and Machine Learning. Human Factors and Ergonomics Society, 2024, Link'), Document(metadata={'Section': 'SKILLS', 'Subsection': ''}, page_content='# SKILLS  \nLanguages: Java, Python, C, C++, HTML/CSS, JavaScript, Bash, Verilog, SQL, TypeScript, Assembly  \nTechnologies: Next.js, React.js, Node.js, Flask, Express.js, AWS, MongoDB, Docker, Machine Learning, Computer Vision, Firebase, LangChain, ChromaDB, ROS2, Git, Tensorflow, Linux, OpenCV, FastAPI, PostgreSQL, OAuth, Supabase, Web Sockets, Github Actions'), Document(metadata={'Section': 'General Details', 'Subsection': ''}, page_content="#I'm Jason, a sophomore studying computer science and engineering at Santa Clara University.\nI'm passionate about being able to make a real-world impact on other people using the skills in software development I have.\nI'm always brainstorming and thinking about new cool projects I can take on, so if anyone has any, they can let me know.\nIn my free time, you can find me playing basketball with my friends or learning something new. However, if I'm not playing basketball, I could also be watching basketball.\nI serve on the board of the Cybersecurity Club on campus at Santa Clara University called BroncoSec. We try to provide knowledge of cybersecurity principles and development practices to those who are interested in the field."), Document(metadata={'Section': 'Nezerac', 'Subsection': ''}, page_content="# Nezerac\nOne of the projects that I developed was during the INRIX X AWS Hackathon in 2024 where myself and other teammates developed an app called Nezerac. Nezerac is an AI agent which helps ease the workload of restaurant owners.  It's use is to make it so closing deals on quality ingredients and other supplies, holding conversations via emails for the restaurant owner, so the owner can focus on serving the customers more than the supply side.\nI specifically worked on communication between the frontend and the backend, in writing and processing restaurant data via a lambda function into dynamodb for the AI Agent."), Document(metadata={'Section': 'Visionairy', 'Subsection': ''}, page_content='# Visionairy\nAnother one of the projects I developed called Visionairy was during Hack for Humanity 2025. In this hackathon, my team secured the Most Likely to Be a Startup prize at a hackathon with over 330+ participants from across California.\nI developed the backend using FastAPI, Langchain, Google Cloud Speech-to-Text Models, and Browser Use to automate and ease the booking process of flights for those who are visually impaired.')]
        self.pool: Pool = None
        
    async def initialize_vector_store(self) -> None:
        # self.clear_collection()
        res = await self.vector_store.aadd_documents(self.docs)
        # print(res)
    
    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.db_url, statement_cache_size=0)
    
    async def query(self, q: str, top_k: int = 3) -> list[Document]:
        try:
            similarity_scores, candidate_docs = await self.query_vector_db(q)
        
            # Rerank these candidate docs using Cohere
            cohere_scores = await self.query_rerank_docs(q, candidate_docs, top_k)
            # 3. Make sure all arrays match in length.
            min_len = min(len(similarity_scores), len(cohere_scores))
            similarity_scores = similarity_scores[:min_len]
            cohere_scores = cohere_scores[:min_len]
            candidate_docs = candidate_docs[:min_len]
            
            # 4. Combine the scores using a lambda weight.
            #    Here, lambda_weight=0.1 means 10% influence from the vector similarity and 90% from the re-ranking.
            lambda_weight = 0.3
            final_scores = lambda_weight * similarity_scores + (1 - lambda_weight) * cohere_scores
            
            # 5. Select the top_k documents based on the combined scores.
            sorted_indices = np.argsort(final_scores)[::-1][:top_k]
            final_docs = [candidate_docs[i] for i in sorted_indices]

            return final_docs
        except Exception as e:
            print(e)
            return []
    
    async def query_vector_db(self, q: str):
        res = await self.vector_store.asimilarity_search_with_relevance_scores(query=q, k=12)
        if not res:
            raise Exception(
                "Problem with the similarity search"
            )
        # print(res)
        similarity_scores = np.array([score for _, score in res])
        similarity_docs = [doc for doc, _ in res]
        similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min() + 1e-8)
        
        return (similarity_scores, similarity_docs)
    
    async def query_rerank_docs(self, q: str, candidate_docs: list[Document], top_k: int):
        keywords = PostgresRAG.extract_keyword(q)
        
        # 1. Track original indices before fuzzy reranking
        indexed_docs = list(enumerate(candidate_docs))  # [(0, doc0), (1, doc1), ...]
        
        # 2. Apply fuzzy reranking while keeping indices
        ranked_indexed_docs = self.rerank_results(
            key_tokens=keywords, 
            indexed_docs=indexed_docs  # Assume this now returns sorted [(orig_idx, doc), ...]
        )
        
        # 3. Extract reranked docs and their original indices
        original_indices = [idx for idx, _ in ranked_indexed_docs]
        ranked_docs = [doc for _, doc in ranked_indexed_docs]
        
        # 4. Get Cohere scores for ALL reranked docs
        doc_texts = [doc.page_content for doc in ranked_docs]
        cohere_ranked_docs = await self.cohere_client.rerank(
            query=q, 
            documents=doc_texts, 
            top_n=len(ranked_docs),  # Score all docs, not just top_k
            model="rerank-v3.5"
        )
        
        # 5. Map Cohere scores back to original indices
        cohere_scores = np.zeros(len(candidate_docs))
        for response in cohere_ranked_docs.results:
            original_idx = original_indices[response.index]  # Key step: align with vector scores
            cohere_scores[original_idx] = response.relevance_score
        
        # 6. Normalize scores
        cohere_scores = (cohere_scores - cohere_scores.min()) / (
            cohere_scores.max() - cohere_scores.min() + 1e-8
        )
        
        return cohere_scores

    @staticmethod
    def extract_keyword(query: str) -> str:
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(query, "english")
        words = [w.lower() for w in words if w.isalnum()]
        keywords = [w for w in words if w not in stop_words]
        
        return set(keywords)
    def rerank_results(self, key_tokens: list[str], indexed_docs: list[tuple[int, Document]]):
        # Calculate fuzzy match score for each doc
        scored_docs = []
        for idx, doc in indexed_docs:
            score = self._fuzzy_match_score(doc.page_content, key_tokens)
            scored_docs.append((idx, doc, score))
        
        # Sort by fuzzy score (descending)
        scored_docs.sort(key=lambda x: x[2], reverse=True)
        
        # Return sorted (index, doc) pairs
        return [(idx, doc) for idx, doc, _ in scored_docs]



    def _fuzzy_match_score(self, document_text: str, keywords: list[str]) -> float:
        """
        Calculate a fuzzy match score between a document's text and a list of keywords.
        
        Args:
            document_text: Text content of the document
            keywords: List of keywords extracted from the query
        
        Returns:
            Normalized score (0-100) indicating match strength
        """
        total_score = 0
        document_text = document_text.lower()
        
        for keyword in keywords:
            # Find best partial match for keyword in document
            score = fuzz.partial_ratio(
                keyword.lower(),
                document_text
            )
            total_score += score
        
        # Average score across all keywords
        return total_score / len(keywords) if keywords else 0

    async def parse_pdf(self, file_name: str) -> bool:
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(BASE_DIR, "../assets/Jason_Wu_Resume.pdf")  # Resolve absolute path
            reader = SimpleDirectoryReader(input_files=[file_path], file_extractor={".pdf": self.parser})
            docs = await reader.aload_data()
            self.docs.extend(docs)
            return True
        except Exception as e:
            raise e
        
    async def parse_markdown(self, file_name: str):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(BASE_DIR, f"../assets/{file_name}")  # Resolve absolute path
        doc_loader = TextLoader(file_path)
        docs = await doc_loader.aload()
        self.docs.extend(docs)
        # await self.chunk_docs(use_c_splitter=True, strip_headers=True)
        # print(docs)
        
    async def clear_table(self) -> None:
        await self.connect()
        async with self.pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE documents;")
            
    
    async def chunk_docs(self, use_c_splitter: bool = False, strip_headers: bool = False) -> None:
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Section"), ("##", "Subsection")], strip_headers=strip_headers)
        new_docs: list[Document] = []
        for doc in self.docs:
            content = doc.page_content or doc.text
            new_content = splitter.split_text(content)
            new_docs.extend(new_content)
        if use_c_splitter:
            c_splitter_docs = self.docs[1:]
            c_splitter = CharacterTextSplitter(separator="\n- ", chunk_size=200, chunk_overlap=50)
            new_docs.extend(c_splitter.split_documents(c_splitter_docs))
        for i in range(len(new_docs)):
            if "# General Details\\n" in new_docs[i].page_content:
                new_docs[i].page_content = new_docs[i].page_content.split("# General Details\\n")[1]
        print("The length of the new docs is", len(new_docs))
        if new_docs:
            new_docs[0].metadata["Section"] = "Contact Information"
        for i in range(len(new_docs)):
            if new_docs[i].metadata.get("Subsection") is None:
                new_docs[i].metadata["Subsection"] = ""
        self.docs = new_docs
        # print(new_docs)
    
    async def close_connection(self):
        if self.pool:
            await self.pool.close()

async def main() -> None:
    db = PostgresRAG()
    # await db.connect()
    # await db.clear_table()
    # db.clear_collection()
    # await db.parse_markdown("extra_info.md")
    # await db.chunk_docs()
    # await db.initialize_vector_store()
    results = await db.query("Where did Jason intern at last year?")
    print(results)
    
    # await db.close_connection()
    
if __name__ == "__main__":
    # db = PostgresRAG()
    asyncio.run(main())
    
    