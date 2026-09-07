"""Line-oriented TCP client for AVPlumber's control server."""

from __future__ import annotations

import asyncio


class AvpProtocolError(RuntimeError):
    pass


class AvpConnection:
    """One serialized connection to AVPlumber's line-oriented TCP protocol."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self.writer is not None

    async def connect(self) -> None:
        await self.disconnect()
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        code, status, _ = await self._read_response()
        if code != 100:
            await self.disconnect()
            raise AvpProtocolError(f"unexpected greeting: {code} {status}")

    async def disconnect(self) -> None:
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
        self.reader = None
        self.writer = None

    async def command(self, command: str) -> str | None:
        async with self._lock:
            if self.reader is None or self.writer is None:
                raise ConnectionError("not connected")
            request = asyncio.create_task(asyncio.wait_for(self._exchange(command), 5.0))
            try:
                return await asyncio.shield(request)
            except asyncio.CancelledError:
                # Finish an already-sent transaction before releasing the lock.
                # A newer TUI worker can then reuse the connection safely.
                try:
                    await request
                except Exception:
                    pass  # _exchange disconnects on transport failure.
                raise

    async def _exchange(self, command: str) -> str | None:
        try:
            self.writer.write(command.encode("utf-8") + b"\n")
            await self.writer.drain()
            code, status, content = await self._read_response()
        except (Exception, asyncio.CancelledError):
            await self.disconnect()
            raise
        if not 200 <= code < 300:
            raise AvpProtocolError(f"{code} {status}")
        return content

    async def _read_response(self) -> tuple[int, str, str | None]:
        if self.reader is None:
            raise ConnectionError("not connected")
        line = await self.reader.readline()
        if not line:
            raise EOFError("AVPlumber closed the control connection")
        code_text, _, status = line.decode("utf-8").rstrip("\n").partition(" ")
        content = None
        if int(code_text) == 201:
            lines = []
            while True:
                line = await self.reader.readline()
                if not line:
                    raise EOFError("AVPlumber closed a multiline response")
                if line in (b"\n", b"\r\n"):
                    break
                lines.append(line.decode("utf-8"))
            content = "".join(lines)
        return int(code_text), status, content
