import asyncio
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent
from typing import AsyncGenerator, Dict, Optional

import asyncpg
import clickhouse_connect
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from scalar_fastapi import get_scalar_api_reference

# Agno imports
from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.text import TextKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.tools.file import FileTools
from agno.tools import tool, Toolkit
from agno.vectordb.pgvector import PgVector

# Local imports
from .sample_queries import sample_queries
from .semantic_model import SEMANTIC_MODEL
from .clickhouse_rules import clickhouse_rules

# --- Configuration & Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Service Configuration
APP_TITLE = os.getenv("APP_TITLE", "Adventure Works ClickHouse Analytics API")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 2))

# ClickHouse Configuration
CLICKHOUSE_HOST = os.getenv("CH_HOST")
CLICKHOUSE_PORT = int(os.getenv("CH_PORT", 8443))
CLICKHOUSE_USERNAME = os.getenv("CH_USERNAME")
CLICKHOUSE_PASSWORD = os.getenv("CH_PASSWORD")
CLICKHOUSE_DATABASE = os.getenv("CH_DATABASE")

# PostgreSQL Configuration (for agent storage and vector db)
POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL")

# AI Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai:gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
cwd = Path(__file__).parent
knowledge_dir = cwd.joinpath("knowledge")
output_dir = cwd.joinpath("output")

# Create directories if they don't exist
knowledge_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# --- Custom ClickHouse Client Wrapper ---
class ClickHouseClient:
    """Thin wrapper around clickhouse_connect client for convenience."""

    def __init__(self):
        self.client = None
        self._connect()

    def _connect(self):
        try:
            self.client = clickhouse_connect.get_client(
                host=CLICKHOUSE_HOST,
                port=CLICKHOUSE_PORT,
                username=CLICKHOUSE_USERNAME,
                password=CLICKHOUSE_PASSWORD,
                database=CLICKHOUSE_DATABASE,
                connect_timeout=30,
                send_receive_timeout=300,
                compress=True,
            )
            logging.info("ClickHouse client connected successfully")
        except Exception as e:
            logging.error(f"Failed to connect to ClickHouse: {e}")
            raise

    def run_sql_query(self, query: str) -> str:
        try:
            if not query or not query.strip():
                return "Error: Empty query provided"
            query = query.rstrip(";")
            result = self.client.query_df(query)
            if result.empty:
                return "Query executed successfully but returned no results."
            if len(result) > 100:
                formatted_result = f"Query returned {len(result)} rows. First 100 rows:\n\n"
                formatted_result += result.head(100).to_string(index=False)
                formatted_result += f"\n\n... and {len(result) - 100} more rows."
            else:
                formatted_result = f"Query returned {len(result)} rows:\n\n" + result.to_string(index=False)
            return formatted_result
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def describe_table(self, table_name: str) -> str:
        try:
            q = f"DESCRIBE TABLE {table_name}"
            result = self.client.query_df(q)
            return f"Table schema for {table_name}:\n\n" + result.to_string(index=False)
        except Exception as e:
            return f"Error describing table: {str(e)}"

    def get_table_sample(self, table_name: str, limit: int = 5) -> str:
        try:
            q = f"SELECT * FROM {table_name} LIMIT {limit}"
            result = self.client.query_df(q)
            return f"Sample data from {table_name}:\n\n" + result.to_string(index=False)
        except Exception as e:
            return f"Error getting table sample: {str(e)}"


# --- Agno Tools for ClickHouse (function-based) ---
# Using function-style tools avoids Toolkit registration complexities
# and the @tool decorator can directly control stop/show behavior.

# Global singleton client for tools
_CLICKHOUSE_CLIENT_SINGLETON = ClickHouseClient()

@tool(show_result=False)
def ch_run_sql_query(query: str) -> str:
    """Execute a ClickHouse SQL query and return a formatted string."""
    return _CLICKHOUSE_CLIENT_SINGLETON.run_sql_query(query)

@tool(show_result=False)
def ch_describe_table(table_name: str) -> str:
    """Describe a ClickHouse table."""
    return _CLICKHOUSE_CLIENT_SINGLETON.describe_table(table_name)
'''
@tool(show_result=True)
def ch_get_table_sample(table_name: str, limit: int = 5) -> str:
    """Return a small sample from a ClickHouse table."""
    return _CLICKHOUSE_CLIENT_SINGLETON.get_table_sample(table_name, limit)
'''

# Security and rate limiting
class RateLimit:
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)

    def is_rate_limited(self, api_key: str) -> tuple[bool, int]:
        now = datetime.now()
        last_minute = now - timedelta(minutes=1)
        self.requests[api_key] = [t for t in self.requests[api_key] if t > last_minute]
        recent = len(self.requests[api_key])
        if recent >= self.requests_per_minute:
            return True, 0
        self.requests[api_key].append(now)
        remaining = self.requests_per_minute - len(self.requests[api_key])
        return False, remaining


rate_limiter = RateLimit(requests_per_minute=60)
header_scheme = APIKeyHeader(name="X-API-Key", auto_error=True)
key = os.getenv("DECODE_KEY")


def verify_api_key(api_key: str = Depends(header_scheme)):
    if api_key != key:
        raise HTTPException(status_code=403, detail="Invalid API Key!")
    is_limited, _ = rate_limiter.is_rate_limited(api_key)
    if is_limited:
        raise HTTPException(status_code=429, detail="Rate limit exceeded!")
    return api_key


# Pydantic API Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="The SQL query or question to analyze")
    session_id: Optional[str] = Field("default", description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    model_id: str = Field(DEFAULT_MODEL, description="AI model to use (format: provider:model_name)")
    stream: bool = Field(True, description="Enable streaming response")
    debug_mode: bool = Field(False, description="Enable debug logging")


class AnalyticsAgent:
    """Analytics Agent with FastAPI integration."""

    def __init__(self):
        # Storage and Knowledge setup
        self.agent_storage = PostgresAgentStorage(
            db_url=POSTGRES_DB_URL,
            table_name="sql_agent_sessions",
            schema="ai",
        )

        self.agent_knowledge = CombinedKnowledgeBase(
            sources=[
                TextKnowledgeBase(path=knowledge_dir, formats=[".txt", ".sql", ".md"]),
                JSONKnowledgeBase(path=knowledge_dir),
            ],
            vector_db=PgVector(
                db_url=POSTGRES_DB_URL,
                table_name="sql_agent_knowledge",
                schema="ai",
                embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            ),
            num_documents=5,
        )

        self.semantic_model_str = json.dumps(SEMANTIC_MODEL, indent=2)
        self.clickhouse_client = _CLICKHOUSE_CLIENT_SINGLETON

    async def initialize_context(self):
        logging.info("Initializing ClickHouse Analytics Agent...")
        await self._create_knowledge_files()
        await asyncio.to_thread(self.agent_knowledge.load)
        logging.info("ClickHouse Analytics Agent initialized successfully.")

    async def _create_knowledge_files(self):
        semantic_file = knowledge_dir / "semantic_model.json"
        with open(semantic_file, "w", encoding="utf-8") as f:
            json.dump(SEMANTIC_MODEL, f, ensure_ascii=False, indent=2)

        samples_file = knowledge_dir / "sample_queries.md"
        with open(samples_file, "w", encoding="utf-8") as f:
            f.write(sample_queries)

        rules_file = knowledge_dir / "clickhouse_rules.md"
        with open(rules_file, "w", encoding="utf-8") as f:
            f.write(clickhouse_rules)

    def _get_model(self, model_id: str):
        try:
            provider, model_name = model_id.split(":")
            if provider == "openai":
                return OpenAIChat(id=model_name)
            else:
                raise ValueError(f"Unsupported model provider: {provider}")
        except ValueError as e:
            logging.error(f"Invalid model_id format: {model_id}. Expected format: 'provider:model_name'")
            raise e

    def get_agent(
        self,
        name: str = "ClickHouse Agent",
        user_id: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
        session_id: Optional[str] = None,
        debug_mode: bool = False,
    ) -> Agent:
        model = self._get_model(model_id)

        # Construct the Agent with tight step limits and without ReasoningTools
        return Agent(
            name=name,
            model=model,
            user_id=user_id,
            session_id=session_id,
            storage=self.agent_storage,
            knowledge=self.agent_knowledge,
            add_history_to_messages=True,
            markdown=True,
            # IMPORTANT: trim auto tools to avoid extra hops
            search_knowledge=True,
            read_chat_history=True,
            read_tool_call_history=False,
            # Register toolkits
            tools=[
                ch_run_sql_query,
                ch_describe_table,
                #ch_get_table_sample,
                FileTools(base_dir=output_dir),
            ],
            show_tool_calls=True,
            #tool_call_limit=3,          # hard cap on tool invocations per run
            reasoning=False,            # no ReasoningTools -> no 'think' loops
            debug_mode=debug_mode,
            description=dedent(
                """\
          You are VAXA (Vaccination Analytics Analyst), an elite Text2SQL Engine specializing in ClickHouse and focusing on:

    - Vaccination coverage and adherence analysis
    - Patient demographics and equity insights
    - Vaccine performance and dose completion metrics
    - Geographic and regional vaccination trends
    - Establishment and healthcare provider performance tracking
    - Campaign and strategy effectiveness evaluation
    - Inventory and lot monitoring for vaccine distribution
    - Data quality, anomaly detection, and reporting completeness

You combine deep public health and epidemiological knowledge with advanced ClickHouse SQL expertise to uncover insights from Brazil’s RNDS vaccination data (vacinas.events)."""
            ),
            instructions=dedent(
                f"""\
            You are an expert in SQL for ClickHouse, focused on writing precise and efficient queries.

When a question requires data, write ONE correct SQL query and execute it via the `run_sql_query` tool.

Steps:
1. Identify the relevant tables in the semantic_model.
2. If necessary, use `describe_table` or `get_table_sample`.
3. **Write a single ClickHouse SQL query** (without semicolon); always include `LIMIT` unless the user asks for everything.
4. Consider nulls and duplicates.
5. Run the query with `run_sql_query`. (The agent should terminate after execution.)
6. Briefly analyze the result and respond in Brazilian Portuguese.

Rules:
- Use ClickHouse syntax (LIMIT, functions like toDate(), formatDateTime()).
- Use backticks on identifiers when necessary.
- Avoid queries without reasonable filters.
- **Never** run write/DDL commands.
- Always show the SQL used to obtain the result.
- Always respond in brazilian portuguese
- Do NOT wrap fully-qualified names in a single pair of backticks.
- Use either db.table (preferred) or `db`.`table`.


            <semantic_model>
            {self.semantic_model_str}
            </semantic_model>\
            """
            ),
        )

    async def process_query(self, request: QueryRequest) -> AsyncGenerator[Dict, None]:
        try:
            agent = self.get_agent(
                user_id=request.user_id,
                model_id=request.model_id,
                session_id=request.session_id,
                debug_mode=request.debug_mode,
            )

            yield {"status": "processing", "message": "Analisando sua consulta..."}

            import concurrent.futures

            def run_agent():
                # Optionally, you can enable stream_intermediate_steps=True for debugging
                return agent.run(request.query)

            yield {"status": "executing", "message": "Processando com IA..."}

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_agent)
                poll_count = 0
                while not future.done():
                    poll_count += 1
                    if poll_count % 2 == 0:
                        yield {"status": "processing", "message": "Ainda processando...", "keepalive": True}
                    await asyncio.sleep(0.3)

                response = future.result()
                reasoning = getattr(response, "content", str(response))

            yield {"status": "analyzing", "message": "Analisando resultados..."}
            await asyncio.sleep(0.01)

            chunk_size = 20
            for i in range(0, len(reasoning), chunk_size):
                chunk = reasoning[i:i + chunk_size]
                yield {
                    "status": "streaming",
                    "content": chunk,
                    "partial_response": reasoning[: i + len(chunk)],
                    "is_final": i + chunk_size >= len(reasoning),
                    "timestamp": datetime.now().isoformat(),
                }
                await asyncio.sleep(0.05)

            yield {
                "status": "completed",
                "reasoning": reasoning,
                "session_id": request.session_id,
                "user_id": request.user_id,
                "model_used": request.model_id,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            yield {"status": "error", "error": str(e), "session_id": request.session_id, "user_id": request.user_id}


analytics_agent = AnalyticsAgent()


async def ensure_env():
    missing = []
    required = ["POSTGRES_DB_URL", "CH_HOST", "CH_USERNAME", "CH_PASSWORD", "CH_DATABASE", "OPENAI_API_KEY"]
    for var in required:
        if not os.getenv(var) or os.getenv(var) in (None, "", "fake"):
            missing.append(var)
    if missing:
        logging.warning(f"Variáveis de ambiente ausentes ou 'fake': {', '.join(missing)}")


async def ensure_clickhouse_ready():
    try:
        temp_client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USERNAME,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DATABASE,
        )
        temp_client.command("SELECT 1")
        logging.info("ClickHouse connection successful")
    except Exception as e:
        logging.warning(f"ClickHouse connection test failed: {e}")
        raise


async def ensure_postgres_ready():
    conn = await asyncpg.connect(dsn=POSTGRES_DB_URL)
    try:
        await conn.execute("CREATE SCHEMA IF NOT EXISTS ai;")
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as ext_err:
            logging.info(f"Não foi possível garantir extensão 'vector' (pgvector): {ext_err}")
    finally:
        await conn.close()


async def warmup_agents():
    await analytics_agent.initialize_context()


console = Console()


@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    console.print(Panel("🚀 Iniciando Servidor", border_style="green"))
    await ensure_env()
    try:
        if POSTGRES_DB_URL and POSTGRES_DB_URL != "fake":
            await ensure_postgres_ready()
        else:
            logging.warning("POSTGRES_DB_URL não configurado corretamente; pulando preparo do Postgres.")

        if all([CLICKHOUSE_HOST, CLICKHOUSE_USERNAME, CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE]):
            await ensure_clickhouse_ready()
        else:
            logging.warning("ClickHouse configuration incomplete; skipping ClickHouse check.")

        await warmup_agents()
        console.print(Panel("✅ Agentes carregados e bancos prontos", border_style="green"))
    except Exception as e:
        logging.exception("Falha no startup/lifespan:")
        logging.info(f"Não foi possível fazer o startup dos bancos e/ou dos agentes: {e}")
    yield
    console.print(Panel("🛑 Parando servidor", border_style="red"))


# FastAPI app
app = FastAPI(title=APP_TITLE, dependencies=[Depends(verify_api_key)], lifespan=lifespan_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining"],
)


@app.post("/sql-query")
async def execute_sql_analytics_query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
    response: Response = None,
):
    """Execute SQL analytics query using Adventure Works ClickHouse agent."""
    _, requests_remaining = rate_limiter.is_rate_limited(api_key)
    if response is not None:
        response.headers["X-RateLimit-Remaining"] = str(requests_remaining)

    try:
        if request.stream:
            # Build a fresh Agent (so per-request flags apply)
            agent = analytics_agent.get_agent(
                user_id=request.user_id,
                model_id=request.model_id,
                session_id=request.session_id,
                debug_mode=request.debug_mode,
            )

            # Agno returns an iterator of RunResponse chunks when stream=True
            run_stream = agent.run(
                request.query,
                stream=True,
                # optional but useful to see tool events/status in the stream
                stream_intermediate_steps=True,
            )

            async def stream_generator():
                # Initial hint for clients
                yield f"data: {json.dumps({'status': 'processing', 'message': 'Analisando sua consulta...'})}\n\n"
                # Iterate over RunResponse chunks (sync iterator; wrap in thread if you prefer)
                for chunk in run_stream:
                    # Each chunk has .event/.content; serialize minimally
                    payload = {
                        "status": "streaming",
                        "event": getattr(chunk, "event", None),
                        "content": getattr(chunk, "content", None),
                        "timestamp": datetime.now().isoformat(),
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                # Finish
                yield "data: {\"status\": \"stream_end\"}\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-RateLimit-Remaining": str(requests_remaining),
                },
            )

        else:
            # Non-streaming path: collect the final RunResponse
            agent = analytics_agent.get_agent(
                user_id=request.user_id,
                model_id=request.model_id,
                session_id=request.session_id,
                debug_mode=request.debug_mode,
            )
            run = agent.run(request.query)  # returns RunResponse
            reasoning = getattr(run, "content", str(run))
            return {
                "status": "completed",
                "reasoning": reasoning,
                "session_id": request.session_id,
                "user_id": request.user_id,
                "model_used": request.model_id,
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        logging.error(f"Error in /sql-query endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.get("/sessions")
async def list_sessions(api_key: str = Depends(verify_api_key)):
    conn = await asyncpg.connect(dsn=POSTGRES_DB_URL)
    try:
        sessions = await conn.fetch(
            """
            SELECT 
                session_id,
                user_id,
                created_at,
                updated_at,
                COUNT(*) as message_count
            FROM ai.sql_agent_sessions 
            GROUP BY session_id, user_id, created_at, updated_at
            ORDER BY updated_at DESC
            """
        )
        session_list = []
        for s in sessions:
            session_list.append({
                "session_id": s["session_id"],
                "user_id": s["user_id"],
                "created_at": s["created_at"] if s["created_at"] else None,
                "updated_at": s["updated_at"] if s["updated_at"] else None,
                "message_count": s["message_count"],
            })
        return {"status": "success", "sessions": session_list, "total": len(session_list)}
    except Exception as e:
        logging.error(f"Error listing sessions: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        await conn.close()


@app.get("/tables")
async def list_available_tables():
    try:
        tables_by_schema = {}
        for table in SEMANTIC_MODEL["tables"]:
            schema_name = table["table_name"].split(".")[0]
            tables_by_schema.setdefault(schema_name, []).append(
                {
                    "name": table["table_name"],
                    "description": table["table_description"],
                    "use_case": table["use_case"],
                }
            )
        return {"status": "success", "total_tables": len(SEMANTIC_MODEL["tables"]), "schemas": tables_by_schema}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/models")
async def list_available_models():
    return {
        "status": "success",
        "models": {"openai": ["gpt-4o", "gpt-4o-mini"]},
        "default_model": DEFAULT_MODEL,
        "format": "provider:model_name",
        "examples": ["openai:gpt-4o", "openai:gpt-4o-mini"],
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, api_key: str = Depends(verify_api_key)):
    conn = await asyncpg.connect(dsn=POSTGRES_DB_URL)
    try:
        result = await conn.execute("DELETE FROM ai.sql_agent_sessions WHERE session_id = $1", session_id)
        rows_affected = int(result.split()[-1]) if result else 0
        return {"status": "success", "message": f"Sessão {session_id} deletada com sucesso", "rows_deleted": rows_affected}
    except Exception as e:
        logging.error(f"Error deleting session: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        await conn.close()


@app.delete("/sessions")
async def delete_all_sessions(api_key: str = Depends(verify_api_key)):
    conn = await asyncpg.connect(dsn=POSTGRES_DB_URL)
    try:
        result = await conn.execute("DELETE FROM ai.sql_agent_sessions")
        rows_affected = int(result.split()[-1]) if result else 0
        return {"status": "success", "message": "Todas as sessões foram deletadas", "rows_deleted": rows_affected}
    except Exception as e:
        logging.error(f"Error deleting all sessions: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        await conn.close()


@app.get("/health")
async def health_check():
    try:
        services = {}
        # ClickHouse
        try:
            if analytics_agent.clickhouse_client.client:
                analytics_agent.clickhouse_client.client.command("SELECT 1")
                services["clickhouse"] = "healthy"
            else:
                services["clickhouse"] = "not_configured"
        except Exception as e:
            services["clickhouse"] = f"unhealthy: {str(e)}"
        # Postgres
        try:
            conn = await asyncpg.connect(dsn=POSTGRES_DB_URL)
            await conn.execute("SELECT 1")
            await conn.close()
            services["postgres"] = "healthy"
        except Exception as e:
            services["postgres"] = f"unhealthy: {str(e)}"

        return {
            "status": "healthy" if all("healthy" in str(v) for v in services.values()) else "degraded",
            "services": services,
            "knowledge_loaded": hasattr(analytics_agent.agent_knowledge, "add_document_to_knowledge_base"),
            "available_tables": len(SEMANTIC_MODEL["tables"]),
            "supported_models": ["openai:gpt-4o", "openai:gpt-4o-mini"],
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/scalar", include_in_schema=False)
def get_scalar_docs():
    return get_scalar_api_reference(openapi_url=app.openapi_url, title=f"{APP_TITLE} Docs")
