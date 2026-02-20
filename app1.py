import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pymongo import MongoClient
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

client = MongoClient(mongo_uri)
db = client["STUDYBOT1"]
collection = db["user"]

llm = ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b")

app1 = FastAPI()

# Fix typo here
app1.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

class ChatRequest(BaseModel):
    user_id: str
    question: str

def get_history(user_id):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = [f"{chat['role']}: {chat['message']}" for chat in chats]
    return "\n".join(history)

@app1.get("/")
def home():
    return {"message": "Welcome to the study bot."}

@app1.post("/chat")
def chat(request: ChatRequest):
    history_str = get_history(request.user_id)

    prompt_text = f"You are a study bot.\nConversation so far:\n{history_str}\nUser: {request.question}\nAssistant:"

    # Invoke LLM
    response = llm.invoke([HumanMessage(content=prompt_text)])
    answer = response.content

    # Save user question
    collection.insert_one({
        "user_id": request.user_id,
        "role": "user",
        "message": request.question,
        "timestamp": datetime.now(timezone.utc)
    })

    # Save bot answer
    collection.insert_one({
        "user_id": request.user_id,
        "role": "assistant",
        "message": answer,
        "timestamp": datetime.now(timezone.utc)
    })

    return {"answer": answer}