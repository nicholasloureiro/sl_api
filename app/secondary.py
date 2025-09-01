# minimal_agent_postgres_memory.py

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.postgres import PostgresStorage               # session/message storage
from agno.memory.v2.memory import Memory                        # memory wrapper
from agno.memory.v2.db.postgres import PostgresMemoryDb         # memory storage

DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# 1) Persist sessions/messages (Agent Storage)
storage = PostgresStorage(
    table_name="agent_sessions",          # will create/use ai.agent_sessions
    db_url=DB_URL,
    auto_upgrade_schema=True              # let Agno create/upgrade the table
)

# 2) Persist user memories (Memory)
memory_db = PostgresMemoryDb(
    table_name="memory",                  # will create/use ai.memory
    db_url=DB_URL
)
memory = Memory(db=memory_db)

# 3) Minimal Agent with both storage and memory enabled
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),   # Agno is model-agnostic; use any supported provider
    storage=storage,                      # <-- sessions & messages saved in Postgres
    memory=memory,                        # <-- user memories saved in Postgres
    enable_user_memories=True,            # create/update memories after each run
    markdown=True
)

if __name__ == "__main__":
    user_id = "user_123"
    session_id = "session_001"

    # First turn
    agent.print_response(
        "Meu nome é Nicholas. Eu gosto de viagens longas de carro.",
        user_id=user_id,
        session_id=session_id,
        stream=True,
    )

    # Second turn (will recall memory + append to persisted session)
    agent.print_response(
        "O que você lembra sobre mim?",
        user_id=user_id,
        session_id=session_id,
        stream=True,
    )
