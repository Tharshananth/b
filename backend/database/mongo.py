"""
MongoDB connection for the orchestrator job database.

This connection is READ-ONLY by contract: the credentials in MONGODB_URI must
belong to a user granted only the `read` role on the job database. Nothing in
this package issues a write, and nothing in it should ever start.
"""
import logging
import os
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError

from config import get_config

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    """Get the shared MongoClient. Built once; pymongo pools internally."""
    global _client

    if _client is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise RuntimeError(
                "MONGODB_URI is not set. The job-status tool cannot start without it."
            )

        cfg = get_config().job_db
        _client = MongoClient(
            uri,
            appname="bsai-chatbot",
            # Every timeout is bounded. A hung Atlas call must never hang a worker.
            serverSelectionTimeoutMS=cfg.server_selection_timeout_ms,
            connectTimeoutMS=cfg.connect_timeout_ms,
            socketTimeoutMS=cfg.socket_timeout_ms,
            maxPoolSize=cfg.max_pool_size,
            retryWrites=False,   # read-only client
            tz_aware=True,
        )
        logger.info("MongoDB client created (db=%s)", cfg.database)

    return _client


def get_job_db() -> Database:
    """Get the orchestrator database handle."""
    return get_mongo_client()[get_config().job_db.database]


def ping() -> bool:
    """Liveness probe for the job database. Never raises."""
    try:
        get_mongo_client().admin.command("ping")
        return True
    except (PyMongoError, RuntimeError) as e:
        logger.error("MongoDB ping failed: %s", e)
        return False


def close_mongo_client() -> None:
    """Close the client on shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB client closed")
