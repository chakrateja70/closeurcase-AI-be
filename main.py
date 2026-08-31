from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from src.core.security import verify_docs_access
from src.routes import api_router

# Docs are served from custom, auth-protected routes below instead of the
# built-in unauthenticated ones.
app = FastAPI(
    title="Closeurcase-AI-Backend",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DocsUser = Annotated[str, Depends(verify_docs_access)]


@app.get("/openapi.json", include_in_schema=False)
async def openapi_schema(_: DocsUser):
    return JSONResponse(
        get_openapi(title=app.title, version=app.version, routes=app.routes)
    )


@app.get("/docs", include_in_schema=False)
async def swagger_ui(_: DocsUser):
    return get_swagger_ui_html(
        openapi_url="/openapi.json", title=f"{app.title} - Swagger UI"
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_ui(_: DocsUser):
    return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return "Closeurcase-AI-Backend is running..."


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
