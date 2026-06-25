from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import classifiers, ingest, krigging, notebooks, render, segmentor

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Tostal Sci-data Platform",
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/v1")
app.include_router(classifiers.router, prefix="/v1")
app.include_router(krigging.router, prefix="/v1")
app.include_router(segmentor.router, prefix="/v1")
app.include_router(render.router, prefix="/v1")
app.include_router(notebooks.router, prefix="/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}