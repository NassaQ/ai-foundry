import asyncio
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logger
from app.core.storage import BlobStorage
from app.core.broker import BaseBroker, AzureServiceBusBroker, RabbitMQBroker
from app.db.cosmos import CosmosClient
from app.services.worker import create_message_handler

_WORKER_ROOT = Path(__file__).resolve().parents[1]

for model_dir in ("ocr-model", "Classification-model"):
    model_path = str(_WORKER_ROOT / model_dir)
    if model_path not in sys.path:
        sys.path.insert(0, model_path)


def get_broker() -> BaseBroker:
    if settings.ENVIRONMENT == "production":
        return AzureServiceBusBroker(settings.MESSAGE_BROKER_URL)
    return RabbitMQBroker(settings.MESSAGE_BROKER_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ocr_pipeline import OCRPipeline

    logger.info("Initializing OCR pipeline (Azure Document Intelligence)...")
    ocr_pipeline = OCRPipeline(
        endpoint=settings.AZURE_DOC_INTELLIGENCE_ENDPOINT,
        api_key=settings.AZURE_DOC_INTELLIGENCE_KEY,
        model_id="prebuilt-layout",
        high_resolution=True,
        locale="ar",
        output_format="markdown",
    )
    logger.info("OCR pipeline ready")

    from azure_openai_classifier import LLMClassifier

    logger.info("Initializing LLM classifier (Azure OpenAI)...")
    classifier = LLMClassifier(
        provider="azure",
        api_key=settings.AZURE_OPENAI_API_KEY,
        endpoint=settings.AZURE_OPENAI_ENDPOINT,
        deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )
    logger.info("LLM classifier ready")

    logger.info("Connecting to message broker...")
    broker = get_broker()
    await broker.connect()
    logger.info("Message broker connected")

    blob = BlobStorage(
        conn_str=settings.BLOB_CONNECTION_STR,
        container=settings.BLOB_STORAGE_CONTAINER_NAME,
    )

    logger.info("Connecting to Cosmos DB...")
    cosmos = CosmosClient(
        conn_str=settings.MONGO_CONNECTION_STR,
        db_name=settings.MONGO_DB_NAME,
        collection=settings.COSMOS_OCR_COLLECTION,
    )
    await cosmos.connect()

    handler = create_message_handler(
        ocr_pipeline=ocr_pipeline,
        classifier=classifier,
        blob=blob,
        cosmos=cosmos,
    )

    logger.info(f"Starting consumer on queue: {settings.AI_FOUNDRY_QUEUE_NAME}")
    consumer_task = asyncio.create_task(
        broker.consume(settings.AI_FOUNDRY_QUEUE_NAME, handler)
    )
    app.state.consumer_task = consumer_task
    logger.info("AI Foundry worker is now listening for messages")

    yield

    logger.info("Shutting down AI Foundry worker...")
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    await broker.close()
    await blob.close()
    await cosmos.close()
    logger.info("AI Foundry worker shut down cleanly")


app = FastAPI(
    title="AI Foundry Worker",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def health():
    task: asyncio.Task | None = getattr(app.state, "consumer_task", None)
    if task and not task.done():
        return {"status": "healthy", "consumer": "running"}
    return JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "consumer": "stopped"},
    )
