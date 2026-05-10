"""
Azure Blob Storage client for the AI Foundry worker.

Supports downloading, uploading, deleting, and folder-checking
operations on blobs for the document processing pipeline.
"""

from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from urllib.parse import unquote, urlparse


class BlobStorage:
    """
    Azure Blob Storage client for the AI Foundry worker.

    Handles:
      - Downloading files for OCR processing
      - Uploading files to organized category folders
      - Deleting original blobs after reorganization
      - Checking if a category folder exists
    """

    def __init__(self, conn_str: str, container: str):
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container = container

    def _extract_name(self, path: str) -> str:
        """Parses the blob name from a URL or returns the raw path."""
        if path.startswith("http"):
            parsed = urlparse(path)
            path_parts = parsed.path.lstrip("/").split("/", 1)
            return (
                unquote(path_parts[1])
                if len(path_parts) > 1
                else unquote(path_parts[0])
            )
        return path

    async def download(self, path: str) -> bytes:
        """Downloads the full content of a blob identified by path or URL."""
        name = self._extract_name(path)
        blob = self.client.get_blob_client(container=self.container, blob=name)
        stream = await blob.download_blob()
        return await stream.readall()

    async def upload(self, data: bytes, path: str) -> str:
        """
        Uploads bytes to the specified blob path.

        Args:
            data: The file content as bytes.
            path: The target blob path (e.g., ``finance/invoice.pdf``).

        Returns:
            The absolute URL of the uploaded blob.
        """
        name = self._extract_name(path)
        blob = self.client.get_blob_client(container=self.container, blob=name)
        await blob.upload_blob(data, overwrite=True)
        return blob.url

    async def delete(self, path: str) -> None:
        """
        Deletes a blob at the specified path.

        Silently ignores if the blob does not exist.
        """
        name = self._extract_name(path)
        blob = self.client.get_blob_client(container=self.container, blob=name)
        try:
            await blob.delete_blob()
        except ResourceNotFoundError:
            pass

    async def folder_exists(self, prefix: str) -> bool:
        """
        Checks if a virtual folder (prefix) exists in the container.

        In Azure Blob Storage, folders are virtual — a folder "exists"
        if there is at least one blob whose name starts with the prefix.

        Args:
            prefix: The folder prefix (e.g., ``finance/``).

        Returns:
            True if at least one blob exists under the prefix.
        """
        container_client = self.client.get_container_client(self.container)
        prefix = prefix.strip("/") + "/"
        async for _ in container_client.list_blobs(name_starts_with=prefix):
            return True
        return False

    async def close(self):
        await self.client.close()
