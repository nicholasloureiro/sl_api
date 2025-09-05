# session.py
from typing import Annotated
from fastapi import Depends, Request
from clickhouse_connect.driver.client import Client


def get_clickhouse_client(request: Request) -> Client:
    return request.app.state.ch_client


ClickHouseDep = Annotated[Client, Depends(get_clickhouse_client)]
