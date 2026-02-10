from dotenv import load_dotenv
import os
import json
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from typing import Any, Dict, List
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


load_dotenv()
wx_api = os.getenv("APIKEY")
project_id = os.getenv("PROJECT_ID")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


# Initialize models and vectorstore
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "abb_faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

llm = LLM(
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    base_url="https://us-south.ml.cloud.ibm.com",
    api_key=wx_api,
    project_id=project_id,
    max_tokens=1500,
    temperature=0.2
)

# --- Custom Tools ---

class QueryCleanerTool(BaseTool):
    name: str = "query_cleaner"
    description: str = "Cleans user queries by removing stop words and noise"
    
    def _run(self, query: str) -> str:
        try:
            # Convert to lowercase and remove special characters
            query = re.sub(r'[^a-zA-Z0-9\s]', '', query.lower())
            
            # Tokenize and remove stop words
            tokens = word_tokenize(query)
            
            filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
            
            return ' '.join(filtered_tokens) if filtered_tokens else query
        except Exception as e:
            return query

class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = "Searches the vector database and returns relevant chunks"
    
    def _run(self, query: str) -> str:
        try:
            docs = vectorstore.similarity_search(query, k=5)
            
            result = "Retrieved Information:\n\n"
            for i, doc in enumerate(docs, 1):
                result += f"Document {i}:\n{doc.page_content}\n\n"
            
            return result
        except Exception as e:
            return f"Error retrieving information: {str(e)}"

class QueryClassifierTool(BaseTool):
    name: str = "query_classifier"
    description: str = "Classifies queries as 'straightforward' or 'needs_clarification'"
    
    def _run(self, query: str) -> str:
        try:
            # Keywords that indicate need for clarification
            clarification_keywords = [
                'help select', 'choose', 'recommend', 'suggest', 'best', 'suitable',
                'which drive', 'what drive', 'drive selection', 'drive recommendation',
                'application', 'my project', 'my system', 'for my', 'help me'
            ]
            
            # Keywords that indicate straightforward queries
            straightforward_keywords = [
                'what is', 'how does', 'explain', 'define', 'specification',
                'datasheet', 'manual', 'installation', 'configuration',
                'troubleshoot', 'error', 'fault', 'maintenance'
            ]
            
            query_lower = query.lower()
            
            # Check for clarification needs
            clarification_score = sum(1 for keyword in clarification_keywords 
                                    if keyword in query_lower)
            
            # Check for straightforward indicators
            straightforward_score = sum(1 for keyword in straightforward_keywords 
                                      if keyword in query_lower)
            
            if clarification_score > straightforward_score and clarification_score > 0:
                return "needs_clarification"
            else:
                return "straightforward"
                
        except Exception as e:
            return "straightforward"  # Default to straightforward if error

class InteractiveQuestionTool(BaseTool):
    name: str = "interactive_question"
    description: str = "Generates the next relevant question based on conversation history and retrieved information"
    
    def _run(self, context: str) -> str:
        try:
            # Enhanced question logic with proper tracking
            base_questions = [
                {
                    "question": "What type of equipment or process will this drive control? (e.g., pump, fan, conveyor, crane, etc.)",
                    "keywords": ["equipment", "process", "pump", "fan", "conveyor", "crane"],
                    "category": "equipment_type"
                },
                {
                    "question": "What is the specific power rating of your motor? (Please provide a number with kW or HP, e.g., '15 kW' or '20 HP')",
                    "keywords": ["power rating", "kw", "hp", "watts", "motor power"],
                    "category": "power_rating"
                },
                {
                    "question": "What voltage level does your system operate at? (Please specify, e.g., 230V, 400V, 690V, etc.)",
                    "keywords": ["voltage", "400v", "690v", "230v", "volts"],
                    "category": "voltage"
                },
                {
                    "question": "What is your operating environment? (e.g., indoor clean area, outdoor, dusty conditions, humid environment, marine application)",
                    "keywords": ["environment", "indoor", "outdoor", "dusty", "humid", "marine"],
                    "category": "environment"
                }
            ]
            
            context_lower = context.lower()
            
            # Check which questions have been asked and answered adequately
            asked_questions = []
            incomplete_answers = []
            
            # Track asked questions from context
            for question_data in base_questions:
                question_asked = False
                answer_adequate = False
                
                # Check if this type of question was asked
                if any(keyword in context_lower for keyword in question_data["keywords"]):
                    question_asked = True
                    
                    # Check if answer is adequate (not just the keyword)
                    if question_data["category"] == "power_rating":
                        # Look for specific power values
                        import re
                        power_match = re.search(r'(\d+(?:\.\d+)?)\s*(kw|hp)', context_lower)
                        answer_adequate = bool(power_match)
                    elif question_data["category"] == "voltage":
                        # Look for specific voltage values
                        voltage_match = re.search(r'(\d+)\s*v', context_lower)
                        answer_adequate = bool(voltage_match)
                    elif question_data["category"] == "equipment_type":
                        # Check if specific equipment mentioned
                        equipment_types = ["pump", "fan", "conveyor", "crane", "compressor", "hoist", "motor"]
                        answer_adequate = any(equip in context_lower for equip in equipment_types)
                    else:
                        # For other categories, check if meaningful answer provided
                        answer_adequate = len([word for word in question_data["keywords"] if word in context_lower]) > 1
                
                if question_asked and not answer_adequate:
                    incomplete_answers.append(question_data)
                elif not question_asked:
                    asked_questions.append(question_data)
            
            # Return first incomplete answer for clarification
            if incomplete_answers:
                return incomplete_answers[0]["question"]
            
            # Return next unasked question
            if asked_questions:
                return asked_questions[0]["question"]
            
            # If all questions asked and answered, signal completion
            return "READY_FOR_RECOMMENDATION"
                
        except Exception as e:
            return "Can you provide more details about your application requirements?"

class FinalRecommendationTool(BaseTool):
    name: str = "final_recommendation"
    description: str = "Generates final recommendation based on all collected information"
    
    def _run(self, conversation_summary: str) -> str:
        return f"GENERATE_FINAL_RECOMMENDATION_BASED_ON: {conversation_summary}"



# --- Agents ---

# Supervisor Agent
supervisor_agent = Agent(
    role='Supervisor',
    goal='Analyze user queries and route them to the appropriate specialist agent',
    backstory='''You are an intelligent supervisor who understands different types of technical queries.
    You can distinguish between straightforward informational questions that can be answered directly
    from the knowledge base, and complex application-specific questions that require gathering more
    details from the user before providing recommendations.''',
    verbose=True,
    allow_delegation=True,
    llm=llm,
    tools=[QueryClassifierTool()]
)

# RAG Agent (for straightforward queries)
rag_agent = Agent(
    role='Technical Documentation Assistant',
    goal='Answer straightforward technical questions using information from the knowledge base',
    backstory='''You are an expert technical assistant who specializes in answering direct questions
    about products, specifications, installation procedures, troubleshooting, and general technical
    information. You synthesize information from multiple sources to provide clear, well-structured
    answers for straightforward queries.''',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[QueryCleanerTool(), VectorSearchTool()]
)

# Enhanced Expert Support Agent (with interactive capabilities)
expert_support_agent = Agent(
    role='Interactive Expert Application Consultant',
    goal='Gather detailed requirements through interactive questioning and provide specialized recommendations',
    backstory='''You are a senior application engineer with deep expertise in drive selection and
    system design. You conduct interactive consultations by asking targeted questions one at a time,
    analyzing each response along with relevant technical documentation, and then asking the next
    most relevant question based on what you've learned. You build comprehensive understanding
    through this iterative process before providing final tailored recommendations.''',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[VectorSearchTool(), InteractiveQuestionTool(), FinalRecommendationTool()]
)

# --- Tasks ---

def create_supervisor_task(user_query: str):
    return Task(
        description=f'''
        Analyze the user query: "{user_query}"
        
        Use the query_classifier tool to determine if this is:
        1. A straightforward query that can be answered directly from documentation
        2. A complex query that needs clarification before providing recommendations
        
        Based on the classification, decide which agent should handle this query:
        - For straightforward queries: delegate to Technical Documentation Assistant
        - For queries needing clarification: delegate to Interactive Expert Application Consultant
        
        Return your decision as: "ROUTE_TO: [agent_name]" followed by the reasoning.
        ''',
        agent=supervisor_agent,
        expected_output="A routing decision with clear reasoning"
    )

def create_rag_task(user_query: str):
    return Task(
        description=f'''
        Answer the user's straightforward question: "{user_query}"
        
        Follow these steps:
        1. Use the query_cleaner tool to clean the query by removing stop words
        2. Use the vector_search tool with the cleaned query to retrieve relevant information
        3. Analyze the retrieved information and synthesize a clear, well-structured answer
        
        IMPORTANT: 
        - Do NOT just copy the retrieved documents
        - Synthesize the information into a coherent response
        - Structure your answer with clear sections
        - Remove duplicate information
        - Focus on directly answering the user's question
        ''',
        agent=rag_agent,
        expected_output="A well-structured, synthesized answer that directly addresses the user's question"
    )

def create_interactive_expert_task(conversation_context: str, stage: str):
    if stage == "ask_question":
        return Task(
            description=f'''
            Based on the conversation context: "{conversation_context}"
            
            Your task is to:
            1. Use the vector_search tool to search for information relevant to the context and previous answers
            2. Use the interactive_question tool with the retrieved information and context to generate the next most relevant question
            3. Present ONLY the question to the user in a clear, friendly manner
            
            Context includes: original query, previous questions asked, and user responses so far.
            
            IMPORTANT: Your response should contain ONLY the next question - no tool details or explanations.
            ''',
            agent=expert_support_agent,
            expected_output="A single, clear, relevant question based on the context and retrieved information"
        )
    
    elif stage == "generate_recommendation":
        return Task(
            description=f'''
            Based on the complete conversation: "{conversation_context}"
            
            Your task is to:
            1. Use the vector_search tool to find the most relevant drive solutions based on all collected requirements
            2. Use the final_recommendation tool to structure your comprehensive recommendation
            3. Provide a detailed recommendation that includes:
               - Summary of the user's requirements
               - Recommended drive model(s) with specifications
               - Key features that match their needs
               - Installation and configuration considerations
               - Any additional recommendations or considerations
            
            Make sure your recommendation is specific, actionable, and directly addresses all the requirements gathered.
            ''',
            agent=expert_support_agent,
            expected_output="A comprehensive, detailed drive recommendation based on all collected requirements"
        )




class InteractiveMultiAgentRAGSystem:
    def __init__(self):
        self.conversation_sessions = {}
    
    def process_query(self, user_input: str, session_id: str = "default"):
        """Process user input through the interactive multi-agent system"""
        
        # Initialize session if it doesn't exist
        if session_id not in self.conversation_sessions:
            self.conversation_sessions[session_id] = {
                'stage': 'initial',
                'agent_assigned': None,
                'original_query': '',
                'conversation_history': [],
                'questions_asked': 0,
                'max_questions': 4,
                'collected_info': {}
            }
        
        session = self.conversation_sessions[session_id]
        
        try:
            # Initial query processing
            if session['stage'] == 'initial':
                return self._handle_initial_query(user_input, session)
            
            # Interactive questioning phase
            elif session['stage'] == 'interactive_questioning':
                return self._handle_interactive_response(user_input, session)
            
            # Final recommendation phase
            elif session['stage'] == 'completed':
                return "The consultation is complete. Type 'reset' to start a new consultation or ask a new question."
                
        except Exception as e:
            return f"Error processing input: {str(e)}"
    
    def _handle_initial_query(self, user_query: str, session: dict):
        """Handle the initial user query and route to appropriate agent"""
        
        session['original_query'] = user_query
        session['conversation_history'].append(f"User Query: {user_query}")
        
        # Use supervisor to classify and route
        supervisor_task = create_supervisor_task(user_query)
        supervisor_crew = Crew(
            agents=[supervisor_agent],
            tasks=[supervisor_task],
            process=Process.sequential,
            verbose=False
        )
        
        routing_result = supervisor_crew.kickoff()
        
        # Parse routing decision
        if "Technical Documentation Assistant" in str(routing_result) or "straightforward" in str(routing_result).lower():
            # Route to RAG agent for direct answer
            rag_task = create_rag_task(user_query)
            crew = Crew(
                agents=[rag_agent],
                tasks=[rag_task],
                process=Process.sequential,
                verbose=False
            )
            result = crew.kickoff()
            session['stage'] = 'completed'
            session['agent_assigned'] = 'rag'
            return str(result)
        
        else:
            # Start interactive questioning with expert agent
            session['stage'] = 'interactive_questioning'
            session['agent_assigned'] = 'expert'
            
            # Generate first question
            return self._ask_next_question(session)
    
    def _handle_interactive_response(self, user_response: str, session: dict):
        """Handle user response during interactive questioning phase"""
        
        # Store the user's response
        session['conversation_history'].append(f"User Response: {user_response}")
        session['questions_asked'] += 1
        
        # Validate if the response is meaningful (not just keywords)
        meaningful_response = self._validate_response(user_response)
        
        if not meaningful_response:
            return "I need more specific information. Could you please provide a more detailed answer?"
        
        # Check if we should ask another question or provide final recommendation
        if session['questions_asked'] >= session['max_questions'] or "that's all" in user_response.lower() or "no more" in user_response.lower():
            return self._generate_final_recommendation(session)
        else:
            return self._ask_next_question(session)
    
    def _validate_response(self, response: str) -> bool:
        """Validate if user response is meaningful and specific"""
        response = response.strip().lower()
        
        # Check for incomplete responses
        if len(response) < 2:
            return False
        
        # Check for just unit responses without values
        if response in ['kw', 'hp', 'v', 'volts', 'watts']:
            return False
            
        # Check for vague responses
        vague_responses = ['above', 'below', 'around', 'approximately', 'something', 'anything']
        if response in vague_responses:
            return False
            
        return True
    
    def _ask_next_question(self, session: dict):
        """Generate and ask the next relevant question"""
        
        # Prepare conversation context
        context = f"Original Query: {session['original_query']}\n"
        context += "\n".join(session['conversation_history'][-10:])  # Last 10 interactions
        
        # Create task for asking next question
        question_task = create_interactive_expert_task(context, "ask_question")
        crew = Crew(
            agents=[expert_support_agent],
            tasks=[question_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        result_str = str(result)
        
        # Check if we should proceed to recommendation
        if "READY_FOR_RECOMMENDATION" in result_str:
            return self._generate_final_recommendation(session)
        
        # Store the question in history
        session['conversation_history'].append(f"Agent Question: {result_str}")
        
        return result_str
    
    def _generate_final_recommendation(self, session: dict):
        """Generate final recommendation based on all collected information"""
        
        session['stage'] = 'generating_recommendation'
        
        # Prepare complete conversation context
        context = f"Original Query: {session['original_query']}\n\n"
        context += "Complete Conversation:\n" + "\n".join(session['conversation_history'])
        
        # Create task for final recommendation
        recommendation_task = create_interactive_expert_task(context, "generate_recommendation")
        crew = Crew(
            agents=[expert_support_agent],
            tasks=[recommendation_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        session['stage'] = 'completed'
        session['conversation_history'].append(f"Final Recommendation: {str(result)}")
        
        return str(result)
    
    def reset_session(self, session_id: str = "default"):
        """Reset conversation state for a session"""
        if session_id in self.conversation_sessions:
            del self.conversation_sessions[session_id]
    
    def get_session_info(self, session_id: str = "default"):
        """Get current session information for debugging"""
        return self.conversation_sessions.get(session_id, {})

# --- Usage Example ---
if __name__ == "__main__":
    rag_system = InteractiveMultiAgentRAGSystem()
    session_id = "user_session_1"
    
    print("🤖 Interactive Multi-Agent RAG System Ready!")
    print("Type 'quit' to exit, 'reset' to reset conversation, 'info' to see session info")
    print("="*70)
    
    while True:
        user_input = input("\nYour input: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.lower() == 'reset':
            rag_system.reset_session(session_id)
            print("Session reset!")
            continue
        elif user_input.lower() == 'info':
            info = rag_system.get_session_info(session_id)
            print(f"Session Info: Stage={info.get('stage', 'N/A')}, Questions Asked={info.get('questions_asked', 0)}")
            continue
        
        if user_input:
            print(f"\n Processing...")
            print("="*50)
            
            try:
                response = rag_system.process_query(user_input, session_id)
                print(f"\nAgent: {response}")
                
            except Exception as e:
                print(f"Error: {e}")
            
            print("="*50)