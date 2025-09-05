# main.py
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.router import router
from app.config import settings

import clickhouse_connect
from clickhouse_connect.driver import httputil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Vaccination Dashboard API starting up...")
    logger.info("API Version: 3.0.0")

    # Cria um PoolManager maior e bloqueante
    pool = httputil.get_pool_manager(
        maxsize=settings.CH_POOL_MAXSIZE,
        num_pools=settings.CH_POOL_NUM_POOLS,
        block=True,
    )

    # Um único client por processo (reutilizado em todas as rotas)
    app.state.ch_client = clickhouse_connect.get_client(
        host=settings.CH_HOST,
        port=settings.CH_PORT,
        username=settings.CH_USERNAME,
        password=settings.CH_PASSWORD,
        database=settings.CH_DATABASE,
        secure=settings.CH_SECURE,          # True para 8443/TLS
        verify=settings.CH_VERIFY,          # verificação de certificado
        pool_mgr=pool,                      # usa o pool customizado
        connect_timeout=settings.CH_CONNECT_TIMEOUT,
        send_receive_timeout=settings.CH_SEND_RECV_TIMEOUT,
        # autogenerate_session_id=False,    # opcional: sessões independentes
    )

    try:
        yield
    finally:
        # Shutdown
        try:
            app.state.ch_client.close()
        except Exception as e:
            logger.warning(f"Error closing CH client: {e}")
        logger.info("Vaccination Dashboard API shutting down...")

app = FastAPI(
    title="Vaccination Analytics Dashboard API",
    description="High-performance analytics API for vaccination data with comprehensive filtering and real-time insights",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Has-More", "X-Query-Time", "X-Cache-Hit"],
)

app.include_router(router=router)
