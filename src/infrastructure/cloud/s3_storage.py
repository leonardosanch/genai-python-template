"""S3 storage adapter — implements StoragePort for AWS S3."""

from typing import Any

from src.domain.exceptions import DataSourceError
from src.domain.ports.storage_port import StoragePort


class S3Storage(StoragePort):
    """Storage adapter for AWS S3 via aiobotocore.

    Args:
        bucket: S3 bucket name.
        region: AWS region.
        endpoint_url: Optional custom endpoint (for MinIO, LocalStack, etc.).
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._region = region
        self._endpoint_url = endpoint_url

    def _session_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"region_name": self._region}
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        return kwargs

    async def read(self, path: str) -> bytes:
        from aiobotocore.session import get_session

        session = get_session()
        async with session.create_client("s3", **self._session_kwargs()) as client:
            try:
                response = await client.get_object(Bucket=self._bucket, Key=path)
                async with response["Body"] as stream:
                    return await stream.read()  # type: ignore[no-any-return]
            except client.exceptions.NoSuchKey as e:
                raise FileNotFoundError(f"S3 key not found: {path}") from e
            except Exception as e:
                raise DataSourceError(f"S3 read error: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        from aiobotocore.session import get_session

        session = get_session()
        async with session.create_client("s3", **self._session_kwargs()) as client:
            try:
                await client.put_object(Bucket=self._bucket, Key=path, Body=data)
            except Exception as e:
                raise DataSourceError(f"S3 write error: {e}") from e

    async def list(self, prefix: str) -> list[str]:
        from aiobotocore.session import get_session

        session = get_session()
        async with session.create_client("s3", **self._session_kwargs()) as client:
            try:
                response = await client.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
                contents = response.get("Contents", [])
                return [obj["Key"] for obj in contents]
            except Exception as e:
                raise DataSourceError(f"S3 list error: {e}") from e

    async def exists(self, path: str) -> bool:
        from aiobotocore.session import get_session

        session = get_session()
        async with session.create_client("s3", **self._session_kwargs()) as client:
            try:
                await client.head_object(Bucket=self._bucket, Key=path)
                return True
            except Exception:
                return False
