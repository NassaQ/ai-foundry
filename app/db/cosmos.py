from motor.motor_asyncio import AsyncIOMotorClient

from app.core.logging import logger


class CosmosClient:
    """Async MongoDB client for Azure Cosmos DB (MongoDB API)."""

    def __init__(self, conn_str: str, db_name: str, collection: str):
        self._conn_str = conn_str
        self._db_name = db_name
        self._collection_name = collection
        self._client: AsyncIOMotorClient | None = None

    async def connect(self):
        self._client = AsyncIOMotorClient(self._conn_str)
        db = self._client[self._db_name]
        self._collection = db[self._collection_name]
        logger.info(
            f"Connected to Cosmos DB: {self._db_name}/{self._collection_name}"
        )

    async def insert_ocr_result(self, doc: dict) -> str:
        """Insert an OCR result document and return the inserted _id as string."""
        result = await self._collection.insert_one(doc)
        return str(result.inserted_id)

    async def close(self):
        if self._client:
            self._client.close()
            self._client = None
