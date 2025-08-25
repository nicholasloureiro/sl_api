import clickhouse_connect
from typing import Annotated
from fastapi import Depends
from app.config import settings


async def get_clickhouse_client():
    """Get ClickHouse client"""
    client = clickhouse_connect.get_client(
        host=settings.CH_HOST,
        port=settings.CH_PORT,  # Use HTTP port for clickhouse-connect
        username=settings.CH_USERNAME,
        password=settings.CH_PASSWORD,
        database=settings.CH_DATABASE
    )
    try:
        yield client
    finally:
        client.close()


ClickHouseDep = Annotated[clickhouse_connect.driver.Client, Depends(get_clickhouse_client)]