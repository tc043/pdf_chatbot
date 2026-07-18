from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

from sqlalchemy.ext.compiler import compiles
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

@compiles(Vector, 'sqlite')
def compile_vector_sqlite(element, compiler, **kw):
    return "TEXT"

@compiles(JSONB, 'sqlite')
def compile_jsonb_sqlite(element, compiler, **kw):
    return "TEXT"


# Create engine
engine_kwargs = {}
if not settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["pool_size"] = 20
    engine_kwargs["max_overflow"] = 10

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    **engine_kwargs
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
