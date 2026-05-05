# Gita Bot API 🙏

Get life guidance from the Bhagavad Gita through a REST API.

## Live API
https://gita-bot-api-11-11.onrender.com

## Endpoints
- GET  /health  — Check server status (public)
- POST /ask     — Ask a question (requires X-API-Key header)
- DELETE /reset — Clear conversation (requires X-API-Key header)

## Tech Stack
- FastAPI + Uvicorn
- LangChain + Groq (llama-3.3-70b)
- FAISS vector store
- HuggingFace embeddings
- Deployed on Render

## Example
POST /ask
Header: X-API-Key: your-key
Body: {"question": "What does Gita say about fear?"}
