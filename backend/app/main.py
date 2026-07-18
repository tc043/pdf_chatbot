import time
import uuid
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.core.database import Base, engine
from app.routers import auth, documents, chat, notes

# Set up structured logging configurations
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Automatically create PostgreSQL tables on startup (simplifies Docker deployment)
try:
    print("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")
except Exception as e:
    print(f"Warning: Database initialization failed (check if Postgres is ready): {str(e)}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for flexible testing, lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID and performance logging middleware
@app.middleware("http")
async def add_correlation_id_and_log(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        ip=request.client.host if request.client else "unknown"
    )
    
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        logger.info("request_completed", status_code=response.status_code, duration_sec=round(duration, 4))
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.exception("request_failed", error=str(e), duration_sec=round(duration, 4))
        return Response(
            content='{"detail": "Internal Server Error"}',
            status_code=500,
            media_type="application/json"
        )

# Register routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(documents.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(notes.router, prefix=settings.API_V1_STR)

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
