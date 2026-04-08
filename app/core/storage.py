from azure.storage.blob.aio import BlobServiceClient
from urllib.parse import unquote, urlparse


class BlobDownloader:
    """
    Minimal Azure Blob Storage client for the AI Foundry worker.
    Only supports downloading files needed for processing.
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

    async def close(self):
        await self.client.close()
