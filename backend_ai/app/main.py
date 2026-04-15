"""
main.py - Entry point FastAPI application
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager
import logging
import uvicorn
from fastapi import FastAPI

from app.api.routes import router, shutdown_pipelines
from app.config import API_HOST, API_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.getLogger(__name__).info("Backend AI khởi động...")
    try:
        yield
    finally:
        shutdown_pipelines()
        logging.getLogger(__name__).info("Backend AI dừng.")

app = FastAPI(
    title="Backend AI - Face Recognition",
    version="1.0.0",
    description="Realtime person tracking + face recognition qua RTSP",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


def run():
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    run()
