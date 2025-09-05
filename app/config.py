from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseSettings):
    CH_HOST: str
    CH_PORT: int = 8123                # HTTP
    CH_USERNAME: str = "default"
    CH_PASSWORD: str = ""
    CH_DATABASE: str = "default"
    CH_SECURE: bool = False            # True se for ClickHouse Cloud (TLS)
    CH_VERIFY: bool = True             # verificação TLS
    CH_POOL_MAXSIZE: int = 32          # conexões por host
    CH_POOL_NUM_POOLS: int = 10        # hosts/rotas distintos
    CH_CONNECT_TIMEOUT: float = 5.0
    CH_SEND_RECV_TIMEOUT: float = 60.0

    model_config = SettingsConfigDict(
        env_file="./.env",
        env_ignore_empty=True,
        extra="ignore",
    )

settings = DatabaseSettings()
