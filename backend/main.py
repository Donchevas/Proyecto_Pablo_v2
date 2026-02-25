import logging
import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field

app = FastAPI(title="API Big Data Academy Chatbot")
logger = logging.getLogger("bigdata_chatbot")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _allowed_origins() -> list[str]:
    configured = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8080,https://proyecto-pablo.vercel.app",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID = os.getenv("PROJECT_ID", "iagen-gcp-cwmi")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

session_store: dict[str, InMemoryChatMessageHistory] = {}


def obtener_modelo() -> ChatVertexAI:
    return ChatVertexAI(
        project=PROJECT_ID,
        model=MODEL_NAME,
        location=LOCATION,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        session_id = request.session_id or str(uuid.uuid4())
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

        if session_id not in session_store:
            history = InMemoryChatMessageHistory()
            system_prompt = (
                "Eres Luciana Molina Condori, asistente de Big Data Academy. "
                "Responde formal pero amigable, usa emojis con moderación y "
                "si preguntan algo fuera de contexto, guía a cursos de Big Data, Cloud o IA."
            )
            history.add_message(SystemMessage(content=system_prompt))
            session_store[session_id] = history

        history = session_store[session_id]
        llm = obtener_modelo()
        history.add_user_message(user_message)
        respuesta_ai = llm.invoke(history.messages)
        history.add_ai_message(respuesta_ai.content)

        return ChatResponse(response=respuesta_ai.content, session_id=session_id)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error procesando /chat")
        raise HTTPException(
            status_code=500,
            detail="Ocurrió un error interno al procesar tu solicitud.",
        ) from exc


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "ok", "service": "Big Data Academy Bot"}
