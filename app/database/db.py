import os
from typing import AsyncIterator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncConnection,
)
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base
import contextlib
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class Database:
    def __init__(self) -> None:
        database_url = os.getenv("SUPABASE_NEW_DB_URL")
        if database_url is None:
            raise ValueError("SUPABASE_NEW_DB_URL environment variable is not set")

        self._engine = create_async_engine(
            database_url,
            echo=True,
            pool_size=10,
            max_overflow=5,
            pool_timeout=30,
            pool_recycle=1800,
        )
        self._sessionmaker = async_sessionmaker(
            autocommit=False, bind=self._engine, autoflush=False, expire_on_commit=False
        )

    @contextlib.asynccontextmanager
    async def connectToDb(self) -> AsyncIterator[AsyncConnection]:
        if self._engine is None:
            raise Exception("Database is not initialized")

        async with self._engine.begin() as connection:
            try:
                yield connection
            except:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def createSession(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None or self._engine is None:
            raise Exception("Unable to obtain database session")

        async with self._sessionmaker.begin() as session:
            try:
                yield session
            except:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self):
        if self._engine is None:
            raise Exception("Database engine not initialized")
        await self._engine.dispose()
        self._engine = None
        self._sessionmaker = None


sessionManager = Database()


async def get_db_connection():
    async with sessionManager.createSession() as session:
        yield session
