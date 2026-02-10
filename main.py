from dotenv import load_dotenv
import os
import re
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from elasticsearch import Elasticsearch
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# --- Load Environment Variables ---
load_dotenv()
wx_api = os.getenv("APIKEY")
project_id = os.getenv("PROJECT_ID")
elasticsearch_url = os.getenv("elasticsearch_url", None)
username = os.getenv("username", None)
password = os.getenv("password", None)

# --- Elasticsearch Connection ---
es_connection = Elasticsearch(
    elasticsearch_url,
    basic_auth=(username, password),
    max_retries=10,
    verify_certs=False,
    retry_on_timeout=True,
    request_timeout=300
)

# --- LLM ---
llm = LLM(
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    base_url="https://us-south.ml.cloud.ibm.com",
    api_key=wx_api,
    project_id=project_id,
    max_tokens=4500
)

# --- Tools ---
class QueryCleanerTool(BaseTool):
    name: str = "query_cleaner"
    description: str = "Cleans user queries by removing stop words and noise, but keeps important short technical words"

    keep_words: list = ["ac", "dc", "hv", "lv", "io", "ip", "hp", "kw"]

    def _run(self, query: str) -> str:
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
        tokens = word_tokenize(query)
        filtered_tokens = [
            word for word in tokens
            if word in self.keep_words or (word not in stop_words and len(word) > 2)
        ]
        return ' '.join(filtered_tokens) if filtered_tokens else query


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = "Searches Elasticsearch vector database and returns only relevant text chunks"

    def _run(self, query: str) -> str:
        search_query = {
            "knn": {
                "field": "vector_query_field.predicted_value",
                "k": 5,
                "num_candidates": 100,
                "query_vector_builder": {
                    "text_embedding": {
                        "model_id": "embaas__sentence-transformers-multilingual-e5-base",
                        "model_text": query
                    }
                }
            }
        }
        response = es_connection.search(index="abb_ingestion", body=search_query)
        texts = [
            hit["_source"].get("text_field", "")
            for hit in response["hits"]["hits"]
            if hit["_source"].get("text_field")
        ]
        return "\n\n".join(texts)


class QueryClassifierTool(BaseTool):
    name: str = "query_classifier"
    description: str = (
        "Classifies queries as either 'rag_agent' (general product info) "
        "or 'expert_support_agent' (application-specific help, sizing, recommendations)."
    )

    def _run(self, query: str) -> str:
        prompt = f"""
        You are an expert supervisor AI.

        Decide the best agent for the query. 

        - If the query is general product information, broad explanations, or overviews:
          → return "rag_agent"
          Examples: 
            - "Tell me about ABB low voltage AC drives"
            - "What is ACS580?"
            - "Explain IP ratings"

        - If the query is asking for help, recommendations, product selection, or calculations:
          → return "expert_support_agent"
          Examples:
            - "Help select a drive for my application"
            - "Recommend a drive for a 15kW pump"
            - "Estimate energy savings with ACS580"
            - "Which drive should I choose for my conveyor?"

        Query:
        "{query}"

        Return only one word: "rag_agent" or "expert_support_agent".
        """
        temp_agent = Agent(
            role='Temp Classifier',
            goal='Route queries to rag_agent or expert_support_agent',
            backstory='Temporary agent for query classification',
            verbose=False,
            allow_delegation=False,
            llm=llm,
            tools=[]
        )
        temp_task = Task(
            description=prompt,
            agent=temp_agent,
            expected_output="rag_agent or expert_support_agent"
        )
        crew = Crew(
            agents=[temp_agent], 
            tasks=[temp_task], 
            process=Process.sequential, 
            verbose=False
        )
        response = crew.kickoff()
        text = str(response).lower()

        if "rag_agent" in text:
            return "rag_agent"
        else:
            return "expert_support_agent"



class QueryClassifierTool2(BaseTool):
    name: str = "expert_query_classifier"
    description: str = (
        "Classifies expert-level queries as either 'straight_forward' "
        "if they contain technical specifications, or 'needs_clarification' if not."
    )

    def _run(self, query: str) -> str:
        prompt = f"""
        You are an expert in analyzing technical queries.

        Your task:
        - If the query contains technical specifications like motor power (kW/HP), voltage (V), drive models (ACS580, ACS880), energy savings, ROI, etc.:
            → return "straight_forward"
        - If the query is general, unclear, or does not contain enough technical details to recommend a drive:
            → return "needs_clarification"

        Query:
        "{query}"

        Return only one word: "straight_forward" or "needs_clarification".
        """
        temp_agent = Agent(
            role='Temp Expert Classifier',
            goal='Classify expert queries for clarification or technical processing',
            backstory='Temporary agent for query classification inside expert support',
            verbose=False,
            allow_delegation=False,
            llm=llm,
            tools=[]
        )
        temp_task = Task(
            description=prompt,
            agent=temp_agent,
            expected_output="straight_forward or needs_clarification"
        )
        crew = Crew(
            agents=[temp_agent], 
            tasks=[temp_task], 
            process=Process.sequential, 
            verbose=False
        )
        response = crew.kickoff()
        text = str(response).lower()

        if "straight_forward" in text:
            return "straight_forward"
        else:
            return "needs_clarification"
        


# --- Agents ---
supervisor_agent = Agent(
    role='Supervisor',
    goal='Analyze user queries and route them to the appropriate agent',
    backstory='Decide if the query has enough technical details for a RAG-based answer',
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[QueryClassifierTool()]
    )

rag_agent = Agent(
    role='Technical Documentation Assistant',
    goal='Answer straightforward technical questions using knowledge base',
    backstory='Expert in technical info, specifications, and troubleshooting',
    verbose=True,
    allow_delegation=False, # allow_delegation=True → The agent is allowed to hand off the task to another agent if it thinks someone else is better suited.
    llm=llm,
    tools=[QueryCleanerTool(),VectorSearchTool()]
)

expert_support_agent = Agent(
    role='Expert Support Agent',
    goal='Provide clarification or technical answers based on query content',
    backstory='This agent either returns a predefined response or uses vector search for queries with specs',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[QueryClassifierTool2(), VectorSearchTool()]
)

# --- Tasks ---
def create_supervisor_task(user_query: str):
    return Task(
        description=f'''
        Analyze the user query: "{user_query}"
        Use query_classifier tool to decide if query is straightforward or needs clarification.
        Return: "ROUTE_TO: [agent_name]" with reasoning.
        ''',
        agent=supervisor_agent,
        expected_output="Routing decision"
    )

def create_rag_task(user_query: str):
    return Task(
        description=f'''
        Answer the user's question: "{user_query}"

        Steps:
        1. Use the query_cleaner tool to clean the query
        2. Use the vector_search tool to retrieve relevant documents
        3. Synthesize a clear, structured answer

        IMPORTANT:
        - Structure the answer into the following sections if needed:
            1. Introduction
            2. Product Range Highlights
            3. Applications and Key Benefits
        - Do NOT just copy-paste retrieved documents
        - Keep the answer concise, informative, and technically accurate
        - Remove duplicates and irrelevant content
        ''',
        agent=rag_agent,
        expected_output="Answer structured with Introduction, Product Range Highlights, Applications and Key Benefits"
    )


def create_support_task(user_query: str):
    return Task(
        description=f'''
        The user asked: "{user_query}"

        Your job:
        - Do NOT try to answer the question.
        - Always respond with the following exact message:

        To help you select the right ABB low voltage AC drive for your application, I’ll need a bit more detail about your use case. Please ask the question including some of the following:

        ### Key Application Details
        1. Type of equipment or process (e.g., pump, fan, conveyor, hoist, tram motor)
        2. Motor power rating (in kW or HP)
        3. Voltage level (e.g., 230V, 400V)
        4. Control requirements (e.g., speed control, torque control, positioning)
        5. Environment (e.g., indoor, outdoor, dusty, humid, marine)
        6. Any specific standards or certifications needed (e.g., marine, ATEX, SIL)
        ''',
        agent=expert_support_agent,
        expected_output="Fixed clarification template only"
    )



class SimpleRAGSystem:
    def process_query(self, user_input: str):
        # Classification step
        classifier_task = Task(
            description=f"Classify the query: '{user_input}'",
            agent=supervisor_agent,
            expected_output="rag_agent or expert_support_agent"
        )
        supervisor_crew = Crew(
            agents=[supervisor_agent],
            tasks=[classifier_task],
            process=Process.sequential,
            verbose=False
        )
        classification = str(supervisor_crew.kickoff()).lower()

        if "rag_agent" in classification:
            rag_task = create_rag_task(user_input)
            rag_crew = Crew(agents=[rag_agent], tasks=[rag_task], process=Process.sequential, verbose=False)
            return str(rag_crew.kickoff())



        # Step 3: Route to Expert Support Agent if application-specific
        elif "expert_support_agent" in classification:
            # Run expert classifier inside expert support agent
            classifier_task2 = Task(
                description=f"Classify expert query: '{user_input}'",
                agent=expert_support_agent,
                expected_output="straight_forward or needs_clarification"
            )
            support_crew = Crew(
                agents=[expert_support_agent],
                tasks=[classifier_task2],
                process=Process.sequential,
                verbose=False
            )
            classification2 = str(support_crew.kickoff()).lower()

            if "straight_forward" in classification2:
                # Query has technical specs → run vector search + expert prompt
                vector_task = Task(
                    description=f'''
                    Answer the user's technical query: "{user_input}"

                    Steps:
                    1. Use the vector_search tool to retrieve relevant chunks
                    2. Synthesize a clear, structured answer referencing drive models, energy savings, ROI, etc.
                    3. Keep answer concise and technically accurate
                    ''',
                    agent=expert_support_agent,
                    expected_output="Expert answer based on technical info"
                )
                vector_crew = Crew(
                    agents=[expert_support_agent],
                    tasks=[vector_task],
                    process=Process.sequential,
                    verbose=False
                )
                return str(vector_crew.kickoff())
            else:

                # Query lacks specs → return clarification template
                return """To help you select the right **ABB low voltage AC drive** for your application, I’ll need a bit more detail about your use case. Could you please share the following?
                ### ⚙️ Key Application Details
                1. **Type of equipment or process** (e.g., pump, fan, conveyor, hoist, tram motor, etc.)
                2. **Motor power rating** (in kW or HP)
                3. **Voltage level** (e.g., 230V, 400V, etc.)
                4. **Control requirements** (e.g., speed control, torque control, positioning)
                5. **Environment** (e.g., indoor, outdoor, dusty, humid, marine)
                6. **Any specific standards or certifications needed** (e.g., marine, ATEX, SIL)

                Once I have this, I can recommend the most suitable drive family (e.g., ACS580, ACS880, ACS380) and configuration options."""



# --- FastAPI App ---
app = FastAPI(title="ABB Multi-Agent RAG API", version="1.0")

rag_system = SimpleRAGSystem()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        response = rag_system.process_query(request.query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "ABB Multi-Agent RAG API is running 🚀"}
