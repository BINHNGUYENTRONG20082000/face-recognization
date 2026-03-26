"""
main.py - Entry point FastAPI application
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import uvicorn
from fastapi import FastAPI

from app.api.routes import router
from app.config import API_HOST, API_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Backend AI - Face Recognition",
    version="1.0.0",
    description="Realtime person tracking + face recognition qua RTSP",
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logging.getLogger(__name__).info("Backend AI khởi động...")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=API_HOST, port=API_PORT, reload=False)
