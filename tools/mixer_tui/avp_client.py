"""Async client for avplumber's line-oriented control protocol."""

from __future__ import annotations

import asyncio
import json
from typing import Optional


class AvpResponse:
    def __init__(self, code: int, status: str, content: Optional[str]):
        self.code = code
        self.status = status
        self.content = content

    def json(self):
        return json.loads(self.content) if self.content else None


class AvpError(RuntimeError):
    pass


class AvpClient:
    """Async TCP client for one avplumber control connection."""

    def __init__(self):
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None

    async def connect(self, host: str, port: int) -> None:
        self.reader, self.writer = await asyncio.open_connection(host, port)
        response = await self.read()
        if response.code != 100:
            raise AvpError(f"Unexpected greeting from avplumber: {response.code} {response.status}")

    async def disconnect(self) -> None:
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
            self.reader = None

    async def read(self, raise_for_status: bool = False) -> AvpResponse:
        if self.reader is None:
            raise EOFError("Not connected to avplumber")
        line = await self.reader.readline()
        line = line.decode("utf-8")
        if not line:
            raise EOFError("Connection closed by avplumber")
        parts = line.split(maxsplit=1)
        code = int(parts[0].strip())
        status = parts[1].strip() if len(parts) > 1 else ""
        content = None
        if code == 201:
            content = ""
            while True:
                line = await self.reader.readline()
                line = line.decode("utf-8")
                if line.rstrip("\n") == "":
                    break
                content += line
        if raise_for_status and not (200 <= code < 300):
            raise AvpError(f"avplumber error {code} {status}")
        return AvpResponse(code=code, status=status, content=content)

    async def write(self, cmd: str) -> None:
        if self.writer is None:
            raise EOFError("Not connected to avplumber")
        self.writer.write(cmd.encode("utf-8") + b"\n")
        await self.writer.drain()

    async def command(self, cmd: str, raise_for_status: bool = True) -> AvpResponse:
        await self.write(cmd)
        return await self.read(raise_for_status=raise_for_status)


class AvpConnection:
    """Shared AVP connection with reconnect and serialized command execution."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._client: Optional[AvpClient] = None
        self._lock = asyncio.Lock()
        self._command_lock = asyncio.Lock()
        self.connected = False

    async def get_client(self) -> Optional[AvpClient]:
        if self._client is not None and self.connected:
            return self._client
        return None

    async def ensure_connected(self) -> bool:
        async with self._lock:
            if self.connected and self._client is not None:
                return True
            client = AvpClient()
            try:
                await client.connect(self.host, self.port)
                self._client = client
                self.connected = True
                return True
            except Exception:
                self._client = None
                self.connected = False
                return False

    async def command(self, cmd: str, raise_for_status: bool = True) -> Optional[AvpResponse]:
        async with self._command_lock:
            client = await self.get_client()
            if client is None:
                return None
            try:
                return await client.command(cmd, raise_for_status=raise_for_status)
            except Exception:
                self.connected = False
                self._client = None
                return None

    async def disconnect(self) -> None:
        if self._client:
            await self._client.disconnect()
        self._client = None
        self.connected = False
