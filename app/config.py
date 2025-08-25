from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    CH_HOST: str
    CH_PORT: int = 9000  # Native port
    CH_USERNAME: str = "default"
    CH_PASSWORD: str = ""
    CH_DATABASE: str = "default"

    model_config = SettingsConfigDict(
        env_file="./.env",
        env_ignore_empty=True,
        extra="ignore",
    )

    @property
    def CH_URL(self):
        """Native protocol URL for CH-driver"""
        return f"clickhouse://{self.CH_USERNAME}:{self.CH_PASSWORD}@{self.CH_HOST}:{self.CH_PORT}/{self.CH_DATABASE}"


settings = DatabaseSettings()