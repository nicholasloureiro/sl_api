from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from app.api.router import router 
from fastapi import FastAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Vaccination Dashboard API starting up...")
    logger.info("API Version: 3.0.0")
    yield
    # Shutdown
    logger.info("Vaccination Dashboard API shutting down...")

app = FastAPI(
    title="Vaccination Analytics Dashboard API",
    description="High-performance analytics API for vaccination data with comprehensive filtering and real-time insights",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Has-More", "X-Query-Time", "X-Cache-Hit"],
)

app.include_router(router=router)
