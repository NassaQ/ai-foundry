import os
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str = "dev"

    # Message Broker
    MESSAGE_BROKER_URL: str = "amqp://guest:guest@localhost:5672/"
    AI_FOUNDRY_QUEUE_NAME: str = "ai_foundry_queue"

    # SQL Server
    SQL_SERVER: str
    SQL_DB_NAME: str
    SQL_USER: str
    SQL_PASS: str
    SQL_DRIVER: str = "ODBC Driver 18 for SQL Server"
    SQL_CONNECT_TIMEOUT: int = 60
    SQL_MAX_RETRIES: int = 3
    SQL_RETRY_DELAY_BASE: int = 2

    # Azure Blob Storage
    BLOB_CONNECTION_STR: str
    BLOB_STORAGE_CONTAINER_NAME: str

    # Azure Document Intelligence (OCR)
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = ""
    AZURE_DOC_INTELLIGENCE_KEY: str = ""

    # Azure OpenAI (Classification)
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4.1-mini"
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"

    # Output
    OUTPUT_DIR: str = "documents"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def SQL_CONNECTION_STRING(self) -> str:
        encoded_pass = quote_plus(self.SQL_PASS)
        return (
            f"mssql+aioodbc://{self.SQL_USER}:{encoded_pass}@"
            f"{self.SQL_SERVER}/{self.SQL_DB_NAME}"
            f"?driver={quote_plus(self.SQL_DRIVER)}"
            "&TrustServerCertificate=yes"
        )


settings = Settings()  # type: ignore

os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
