import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from mem0 import Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()


POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": POSTGRES_HOST,
            "port": int(POSTGRES_PORT),
            "dbname": POSTGRES_DB,
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "collection_name": POSTGRES_COLLECTION_NAME,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
        # Optional Feinsteuerung für Graph-Extraktion
        "custom_prompt": "Extrahiere ausschließlich stabile, nützliche Entitäten/Beziehungen (Geräte/Rooms/Services/Capabilities); kein Chat‑Meta. Dedupliziere (slugify/lowercase), mappe Selbstreferenzen auf den aktuellen Nutzer, und lege nur Beziehungen für explizit genannte Entitäten an.",
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "temperature": 0.2, "model": "gpt-4o-mini"}},
    "embedder": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"}},
    "history_db_path": HISTORY_DB_PATH,
    # Custom Fact Extraction Prompt (strikt JSON: {"facts": [string,...]})
    "custom_fact_extraction_prompt": (
        "Du bist der Memory‑Manager. Extrahiere ausschließlich langlebige, nützliche Informationen aus der Eingabe. "
        "Keine Floskeln, Entschuldigungen, Selbstbeschreibungen, kein Chat‑Meta und keine Zusammenfassungen. "
        "Speichere nur, was später nachweislich hilft (z. B. stabile Fakten, klare Vorlieben/Defaults, kanonische Entitäten/IDs, wiederkehrende Routinen). "
        "Dedupliziere semantisch (nur neue oder präzisere Inhalte). Nutze bei Selbstreferenzen (\"ich\", \"mir\", \"mein\") den aktuellen Nutzer. "
        "Halte die Sprache der Eingabe bei. Normalisiere Datumsangaben in YYYY‑MM‑DD und nutze kanonische IDs (z. B. light.bedroom). "
        "Liefere höchstens 5 kurze, präzise Sätze; wenn nichts qualifiziert, gib eine leere Liste zurück. "
        "Antworte ausschließlich als JSON‑Objekt mit dem Schlüssel facts (Liste von Strings). Keine weitere Ausgabe, keine Codeblöcke. "
        "Beispiele: Input: Hi -> {\"facts\":[]}; Input: Ich heiße René. -> {\"facts\":[\"Der Nutzer heißt René.\"]}; "
        "Input: Meine Schlafzimmer‑Lampe heißt light.bedroom. -> {\"facts\":[\"Die Schlafzimmer‑Lampe hat die ID light.bedroom.\"]}"
    ),
    # Custom Update Memory Prompt (strikt JSON: {"memory": [{event,text,id?}]})
    "custom_update_memory_prompt": (
        "Du entscheidest, wie bestehende Erinnerungen anhand neuer Informationen aktualisiert werden. "
        "Es gibt vier Ereignisse: ADD (neue Erinnerung), UPDATE (bestehende Erinnerung mit id aktualisieren), DELETE (bestehende Erinnerung mit id löschen), NONE (keine Änderung). "
        "Aktualisiere nur, wenn die neue Information präziser, aktueller oder korrigierend ist; vermeide Duplikate (bei hoher Ähnlichkeit bevorzuge UPDATE statt ADD). "
        "Verwende für UPDATE/DELETE ausschließlich die bereitgestellten bestehenden ids; bei ADD darf eine neue id erzeugt werden. "
        "Jeder Eintrag enthält ein knappes Feld text (eine klare Ein‑Satz‑Formulierung). Erzeuge höchstens 5 Aktionen; wenn nichts qualifiziert, liefere eine leere Liste. "
        "Antworte ausschließlich im geforderten JSON‑Format; kein zusätzlicher Text, keine Codeblöcke. "
        "Liefere keine \"NONE\"‑Einträge, wenn insgesamt nichts zu tun ist – dann {\\\"memory\\\":[]} zurückgeben."
    ),
}


MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.0.0",
)


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Optional flags to influence extraction behavior
    infer: Optional[bool] = Field(
        default=None,
        description="Whether to use LLM for fact extraction. If None, server default is used.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Optional custom prompt to use for memory creation.",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


@app.post("/configure", summary="Configure Mem0")
def set_config(config: Dict[str, Any]):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}


@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = MEMORY_INSTANCE.add(
            messages=[m.model_dump() for m in memory_create.messages], **params
        )
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str):
    """Retrieve a specific memory by ID."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return MEMORY_INSTANCE.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: Dict[str, Any]):
    """Update an existing memory with new content.
    
    Args:
        memory_id (str): ID of the memory to update
        updated_memory (str): New content to update the memory with
        
    Returns:
        dict: Success message indicating the memory was updated
    """
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str):
    """Retrieve memory history."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory():
    """Completely reset stored memories."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")
