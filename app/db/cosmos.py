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
        await self._collection.create_index("doc_id", unique=True)
        logger.info(
            f"Connected to Cosmos DB: {self._db_name}/{self._collection_name}"
        )

    async def insert_ocr_result(self, doc: dict) -> str:
        """Insert an OCR result document and return the inserted _id as string."""
        result = await self._collection.insert_one(doc)
        return str(result.inserted_id)

    async def find_by_doc_id(self, doc_id: int) -> dict | None:
        """Find an OCR result document by its SQL doc_id."""
        return await self._collection.find_one({"doc_id": doc_id})

    async def upsert_ocr_result(self, doc: dict) -> str:
        """Insert or replace an OCR result document by doc_id. Returns the _id as string."""
        result = await self._collection.replace_one(
            {"doc_id": doc["doc_id"]}, doc, upsert=True
        )
        if result.upserted_id:
            return str(result.upserted_id)
        existing = await self.find_by_doc_id(doc["doc_id"])
        return str(existing["_id"]) if existing else ""

    async def delete_by_doc_id(self, doc_id: int) -> bool:
        """Delete an OCR result document by its SQL doc_id. Returns True if deleted."""
        result = await self._collection.delete_one({"doc_id": doc_id})
        return result.deleted_count > 0

    async def close(self):
        if self._client:
            self._client.close()
            self._client = None
