"""Async database engine and session factory for the ledger subsystem.

Wraps SQLAlchemy's async engine and session machinery in a simple lifecycle
object.  Callers obtain sessions via :meth:`Database.session` and are
responsible for committing or rolling back within their own ``async with``
block.

Usage::

    db = Database("sqlite+aiosqlite:///trading.db")
    await db.init()

    async with db.session() as session:
        session.add(SomeRecord(...))
        await session.commit()

    await db.close()
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ledger.models import Base

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class Database:
    """Lifecycle manager for the async SQLAlchemy engine and session factory.

    Args:
        url: SQLAlchemy async connection URL, e.g.
            ``"sqlite+aiosqlite:///trading.db"`` for development or
            ``"postgresql+asyncpg://user:pass@host/db"`` for production.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def init(self) -> None:
        """Create the async engine, session factory, and all ORM tables.

        Idempotent: calling ``init()`` more than once on an already-initialised
        instance is safe — it is a no-op after the first call.

        Raises:
            sqlalchemy.exc.OperationalError: If the database URL is
                unreachable or the driver is missing.
        """
        if self._engine is not None:
            return

        self._engine = create_async_engine(
            self._url,
            # Echo SQL only at DEBUG level; controlled by structlog elsewhere.
            echo=False,
            # Reasonable pool size for an async trading system.
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        log.info("database.init", url=self._url)

    async def close(self) -> None:
        """Dispose the connection pool and release all resources.

        Safe to call multiple times; subsequent calls after the first are
        no-ops.
        """
        if self._engine is None:
            return
        await self._engine.dispose()
        self._engine = None
        self._session_factory = None
        log.info("database.closed")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a new :class:`~sqlalchemy.ext.asyncio.AsyncSession`.

        Commits on clean exit and propagates exceptions to the caller so
        they can roll back if needed.  Intended for use with ``async with``.

        Yields:
            A fresh :class:`AsyncSession` bound to the engine.

        Raises:
            RuntimeError: If :meth:`init` has not been called yet.

        Example::

            async with db.session() as sess:
                sess.add(record)
                # commit is called automatically on exit
        """
        if self._session_factory is None:
            raise RuntimeError(
                "Database.init() must be called before creating sessions."
            )
        async with self._session_factory() as sess:
            yield sess
            await sess.commit()
