from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from datetime import datetime
import time
import uvicorn
import os

load_dotenv()

# --- Knowledge Base ---
print("Building knowledge base...")

GITA_TEXT = """
The Bhagavad Gita is a 700 verse Hindu scripture that is part of the epic Mahabharata.
It contains a conversation between Pandava prince Arjuna and his guide Lord Krishna.
Arjuna is filled with doubt before the Kurukshetra war. Krishna counsels Arjuna on duty,
righteousness, devotion, and the nature of the soul.

Chapter 2 - Sankhya Yoga: Krishna explains that the soul is eternal and cannot be 
destroyed. He introduces Nishkama Karma - performing duty without attachment to results.
You have a right to perform your duties but you are not entitled to the fruits of action.

Chapter 3 - Karma Yoga: Krishna explains the path of selfless action. One must 
perform their duties without selfish desires. Inaction is not an option for anyone.

Chapter 6 - Dhyana Yoga: The mind is the best friend of those who have controlled it,
and the worst enemy of those who have not.

Chapter 12 - Bhakti Yoga: Krishna declares that devotion is the highest path.
Those who worship him with faith and devotion are most dear to him.

Chapter 18 - Moksha Yoga: Abandon all duties and take refuge in him alone.
He will liberate you from all sins.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.create_documents([GITA_TEXT])
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print("Knowledge base ready.\n")

# --- LLM ---
llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a wise Bhagavad Gita assistant.
    Use the following Gita teachings to answer with wisdom and compassion.
    Relevant teachings: {context}
    Be concise. Maximum 3 paragraphs."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = prompt | llm

# --- API Key Setup ---
# Read from environment instead of hardcoding
API_KEY_1 = os.getenv("API_KEY_1", "")
API_KEY_2 = os.getenv("API_KEY_2", "")
VALID_API_KEYS = {key for key in [API_KEY_1, API_KEY_2] if key}
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Include X-API-Key header."
        )
    return api_key

# --- App ---
app = FastAPI(
    title="Bhagavad Gita Bot API",
    description="Secure production-grade Gita guidance API",
    version="3.0.0"
)

# --- CORS Middleware ---
# This tells browser which origins are allowed to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # local frontend development
        "http://localhost:8080",    # another common dev port
        "https://yourdomain.com",   # your production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# --- State ---
conversation_history = []
request_count = 0
start_time = datetime.now()

# --- Models ---
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

class AskResponse(BaseModel):
    answer: str
    status: str
    response_time_ms: float

class HealthResponse(BaseModel):
    status: str
    message: str
    uptime_seconds: float
    total_requests: int

# --- Endpoints ---

# Health is PUBLIC — no auth needed
@app.get("/health", response_model=HealthResponse)
def health_check():
    uptime = (datetime.now() - start_time).total_seconds()
    return {
        "status": "ok",
        "message": "Gita Bot API is running. Jai Shri Jagannath!",
        "uptime_seconds": round(uptime, 2),
        "total_requests": request_count
    }

# Ask is PROTECTED — requires valid API key
# Notice: depends=Depends(verify_api_key) — this runs before the function
@app.post("/ask", response_model=AskResponse)
def ask_gita(request: AskRequest, api_key: str = Depends(verify_api_key)):
    global conversation_history, request_count

    request_count += 1
    start = time.time()

    try:
        docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in docs])

        response = chain.invoke({
            "context": context,
            "chat_history": conversation_history,
            "question": request.question
        })

        answer = response.content
        response_time = round((time.time() - start) * 1000, 2)

        conversation_history.append(HumanMessage(content=request.question))
        conversation_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "status": "success",
            "response_time_ms": response_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bot error: {str(e)}")

# Reset is also PROTECTED
@app.delete("/reset")
def reset(api_key: str = Depends(verify_api_key)):
    global conversation_history, request_count
    conversation_history = []
    request_count = 0
    return {"status": "ok", "message": "Reset successful"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)