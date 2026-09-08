import logging
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.exceptions import (
    BaseAPIException,
    base_api_exception_handler,
    http_exception_handler,
    rate_limit_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from src.core.logging_config import configure_logging
from src.core.rate_limit import limiter
from src.core.request_context import RequestContextMiddleware
from src.core.security import verify_docs_access
from src.routes import api_router
from src.services.case_detection_service import CaseDetectionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the lifecycle of anything holding a connection pool.

    Services are built here rather than at module import so that importing a
    route module has no side effects, and so the OpenAI client is created with
    a running event loop underneath it. The matching `aclose` on shutdown is
    what stops each reload from leaking the pool.
    """
    configure_logging()
    app.state.case_detection_service = CaseDetectionService()
    logger.info("startup complete")
    try:
        yield
    finally:
        await app.state.case_detection_service.aclose()
        logger.info("shutdown complete")


# Docs are served from custom, auth-protected routes below instead of the
# built-in unauthenticated ones.
app = FastAPI(
    title="Closeurcase-AI-Backend",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# slowapi reads the limiter off app.state; the middleware is what applies the
# limits declared by @limiter.limit on individual routes.
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Added last => outermost, so it sees every request including the ones
# SlowAPIMiddleware rejects, and its timing covers the whole stack.
app.add_middleware(RequestContextMiddleware)

# Every error path returns the same flat envelope as a success response.
# Order matters only in that the most specific class must be registered:
# Starlette resolves handlers by exact class first, then walks the MRO.
app.add_exception_handler(BaseAPIException, base_api_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

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
