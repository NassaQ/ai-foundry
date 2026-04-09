import asyncio
import os
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.exc import OperationalError

from app.core.logging import logger
from app.core.storage import BlobDownloader
from app.core.config import settings
from app.db.cosmos import CosmosClient
from app.db.session import AsyncSessionLocal
from app.models.models import Documents, ProcessingStatus, OcrResult
from app.services.ocr_utils import strip_markdown

from sqlalchemy import select


async def db_operation_with_retry(operation: Callable, *args, **kwargs):
    """
    Execute an async DB operation with exponential backoff retry.
    Catches sqlalchemy OperationalError (connection resets, timeouts)
    and retries up to SQL_MAX_RETRIES times before re-raising.
    """
    last_exception = None

    for attempt in range(1, settings.SQL_MAX_RETRIES + 1):
        try:
            return await operation(*args, **kwargs)
        except OperationalError as e:
            last_exception = e
            if attempt < settings.SQL_MAX_RETRIES:
                delay = settings.SQL_RETRY_DELAY_BASE ** attempt
                logger.warning(
                    f"DB operation failed (attempt {attempt}/{settings.SQL_MAX_RETRIES}). "
                    f"Retrying in {delay}s... Error: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"DB operation failed after {settings.SQL_MAX_RETRIES} attempts. Error: {e}"
                )

    raise last_exception


async def _update_status_inner(doc_id: int, status: str, error_message: str | None = None):
    async with AsyncSessionLocal() as session:
        query = select(ProcessingStatus).where(
            ProcessingStatus.doc_id == doc_id,
            ProcessingStatus.stage_name == "OCR",
        )
        record = (await session.execute(query)).scalar_one_or_none()

        if not record:
            logger.error(f"No ProcessingStatus record found for doc_id={doc_id}")
            return

        record.status = status

        if status == "Processing":
            record.start_time = datetime.now(timezone.utc)
        elif status in ("Finished", "Failed"):
            record.end_time = datetime.now(timezone.utc)

        if error_message:
            record.error_message = error_message

        await session.commit()


async def update_status(doc_id: int, status: str, error_message: str | None = None):
    await db_operation_with_retry(_update_status_inner, doc_id, status, error_message)


async def _update_mongo_doc_id_inner(doc_id: int, mongo_doc_id: str):
    async with AsyncSessionLocal() as session:
        query = select(Documents).where(Documents.doc_id == doc_id)
        doc = (await session.execute(query)).scalar_one_or_none()

        if doc:
            doc.mongo_doc_id = mongo_doc_id
            await session.commit()


async def update_mongo_doc_id(doc_id: int, mongo_doc_id: str):
    await db_operation_with_retry(_update_mongo_doc_id_inner, doc_id, mongo_doc_id)


async def _insert_ocr_result_sql_inner(
    doc_id: int,
    ocr_result,
    classification_result,
):
    async with AsyncSessionLocal() as session:
        row = OcrResult(
            doc_id=doc_id,
            page_count=ocr_result.page_count,
            word_count=ocr_result.word_count,
            avg_confidence=ocr_result.avg_confidence,
            primary_language=ocr_result.primary_language,
            category=classification_result.category if classification_result else None,
            classification_confidence=classification_result.confidence if classification_result else None,
            cost_usd_ocr=ocr_result.cost_usd,
            cost_usd_classification=classification_result.cost_usd if classification_result else None,
            processed_at=datetime.now(timezone.utc),
        )
        session.add(row)
        await session.commit()


async def insert_ocr_result_sql(doc_id: int, ocr_result, classification_result):
    await db_operation_with_retry(
        _insert_ocr_result_sql_inner, doc_id, ocr_result, classification_result
    )


async def process_document(
    message: dict,
    ocr_pipeline,
    classifier,
    blob: BlobDownloader,
    cosmos: CosmosClient,
):
    """
    Process a single queue message: download from blob, run OCR + classification,
    store results in Cosmos DB and SQL Server.
    """
    doc_id = message["doc_id"]
    file_path = message["file_path"]
    filename = message["filename"]

    logger.info(f"Processing doc_id={doc_id}, filename={filename}")

    await update_status(doc_id, "Processing")

    try:
        file_content = await blob.download(file_path)
        logger.info(f"Downloaded {len(file_content)} bytes from blob for doc_id={doc_id}")

        logger.info(f"Running OCR for doc_id={doc_id}")
        ocr_result = ocr_pipeline.run_bytes(file_content, filename=filename)

        if not ocr_result.success:
            raise RuntimeError(f"OCR failed: {ocr_result.error}")

        extracted_text = ocr_result.cleaned_text
        plain_text = strip_markdown(extracted_text)

        classification_result = None
        if plain_text.strip():
            logger.info(f"Running classification for doc_id={doc_id}")
            classification_result = classifier.classify(plain_text)
            logger.info(
                f"Classification for doc_id={doc_id}: "
                f"{classification_result.category} ({classification_result.confidence:.0%})"
            )

        _, ext = os.path.splitext(filename)
        cosmos_doc = {
            "doc_id": doc_id,
            "filename": filename,
            "file_type": ext.lower(),
            "blob_path": file_path,
            "extracted_text": extracted_text,
            "ocr": {
                "page_count": ocr_result.page_count,
                "word_count": ocr_result.word_count,
                "avg_confidence": ocr_result.avg_confidence,
                "primary_language": ocr_result.primary_language,
                "cost_usd": ocr_result.cost_usd,
                "elapsed_seconds": ocr_result.elapsed_seconds,
                "chunks_used": ocr_result.chunks_used,
            },
            "classification": None,
            "processed_at": datetime.now(timezone.utc),
        }

        if classification_result:
            cosmos_doc["classification"] = {
                "category": classification_result.category,
                "confidence": classification_result.confidence,
                "reasoning": classification_result.reasoning,
                "tokens_used": classification_result.tokens_used,
                "cost_usd": classification_result.cost_usd,
                "error": classification_result.error,
            }

        inserted_id = await cosmos.insert_ocr_result(cosmos_doc)
        logger.info(f"Inserted Cosmos document {inserted_id} for doc_id={doc_id}")

        await update_mongo_doc_id(doc_id, inserted_id)
        await insert_ocr_result_sql(doc_id, ocr_result, classification_result)
        await update_status(doc_id, "Finished")

        logger.info(f"Finished processing doc_id={doc_id}")

    except Exception as e:
        logger.error(f"Failed to process doc_id={doc_id}: {e}")
        await update_status(doc_id, "Failed", error_message=str(e))
        raise


def create_message_handler(
    ocr_pipeline,
    classifier,
    blob: BlobDownloader,
    cosmos: CosmosClient,
):
    """
    Factory that creates the message callback with access to the
    OCR pipeline, classifier, blob client, and Cosmos client.
    Returns an async callback suitable for broker.consume().
    """

    async def handle_message(message: dict):
        try:
            await process_document(message, ocr_pipeline, classifier, blob, cosmos)
        except Exception as e:
            logger.error(
                f"Message handler caught error for doc_id={message.get('doc_id')}: {e}"
            )
            raise

    return handle_message
