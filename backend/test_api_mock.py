import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Override environment before importing database or settings
os.environ["DATABASE_URL"] = "sqlite:///test_suite.db"
os.environ["JWT_SECRET"] = "mock-secret-key-for-testing"
os.environ["S3_ENDPOINT"] = "http://mock-s3"
os.environ["REDIS_URL"] = "redis://mock-redis"

from sqlalchemy.ext.compiler import compiles
from pgvector.sqlalchemy import Vector

# 2. SQLite does not support pgvector's Vector type.
# We map it to TEXT dynamically for SQLite compilation.
@compiles(Vector, 'sqlite')
def compile_vector_sqlite(element, compiler, **kw):
    return "TEXT"

from sqlalchemy.dialects.postgresql import JSONB

@compiles(JSONB, 'sqlite')
def compile_jsonb_sqlite(element, compiler, **kw):
    return "TEXT"

# Mock Celery delay before importing router modules that use it
from celery import Celery
Celery.delay = MagicMock()

# Mock external ML libraries to prevent long download/import times during test runtime
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()

from fastapi.testclient import TestClient
from app.core.config import settings
from app.core.database import Base, engine, SessionLocal
from app.services.storage import storage_service
from app.main import app

class TestEnterpriseAPIMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create SQLite tables in memory
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)
        
        # Test accounts
        cls.test_email = "testuser@enterprise.com"
        cls.test_pass = "securepassword123"
        cls.token = ""

    @classmethod
    def tearDownClass(cls):
        engine.dispose()
        if os.path.exists("test_suite.db"):
            try:
                os.remove("test_suite.db")
            except Exception:
                pass

    def test_01_user_registration(self):
        """
        Verifies that a user can register successfully.
        """
        response = self.client.post(
            f"{settings.API_V1_STR}/auth/register",
            json={"email": self.test_email, "password": self.test_pass}
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["email"], self.test_email)
        self.assertTrue(data["is_active"])
        self.assertIn("id", data)

    def test_02_duplicate_user_registration_fails(self):
        """
        Verifies that registering an existing email fails.
        """
        response = self.client.post(
            f"{settings.API_V1_STR}/auth/register",
            json={"email": self.test_email, "password": self.test_pass}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("already exists", response.json()["detail"])

    def test_03_user_login(self):
        """
        Verifies that login returns a valid JWT token.
        """
        response = self.client.post(
            f"{settings.API_V1_STR}/auth/login",
            data={"username": self.test_email, "password": self.test_pass}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["token_type"], "bearer")
        self.assertIn("access_token", data)
        # Save token for subsequent tests
        self.__class__.token = data["access_token"]

    def test_04_get_current_user_profile(self):
        """
        Verifies profile lookup using Bearer JWT.
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        response = self.client.get(
            f"{settings.API_V1_STR}/auth/me",
            headers=headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["email"], self.test_email)

    def test_05_unauthorized_profile_lookup_fails(self):
        """
        Verifies profile lookup fails without JWT.
        """
        response = self.client.get(f"{settings.API_V1_STR}/auth/me")
        self.assertEqual(response.status_code, 401)

    @patch("app.routers.documents.process_pdf_task")
    def test_06_document_upload_flow(self, mock_celery_task):
        """
        Verifies uploading a PDF files triggers background processing.
        """
        # Mock S3 upload call
        storage_service.upload_file = MagicMock(return_value="mock-s3-key")
        
        # Build mock PDF bytes
        import io
        import fitz  # PyMuPDF
        
        # Create a simple 1-page PDF using PyMuPDF to pass page verification
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.write()
        doc.close()
        
        pdf_file = io.BytesIO(pdf_bytes)
        
        headers = {"Authorization": f"Bearer {self.token}"}
        response = self.client.post(
            f"{settings.API_V1_STR}/documents/upload?model_type=OpenAI",
            headers=headers,
            files={"file": ("test_report.pdf", pdf_file, "application/pdf")}
        )
        
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertEqual(data["filename"], "test_report.pdf")
        self.assertEqual(data["status"], "processing")
        self.assertEqual(data["page_count"], 1)
        
        # Verify background task was dispatched
        mock_celery_task.delay.assert_called_once()
        
        # Store document ID for conversational tests
        self.__class__.doc_id = data["id"]

    def test_07_list_documents(self):
        """
        Verifies listing uploaded documents.
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        response = self.client.get(
            f"{settings.API_V1_STR}/documents/",
            headers=headers
        )
        self.assertEqual(response.status_code, 200)
        docs = response.json()
        self.assertTrue(len(docs) > 0)
        self.assertEqual(docs[0]["filename"], "test_report.pdf")

    def test_08_create_conversation_thread(self):
        """
        Verifies creating a chat conversation thread linked to a document.
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        response = self.client.post(
            f"{settings.API_V1_STR}/chats/conversations",
            headers=headers,
            json={"title": "Q3 Growth Chat", "document_id": self.doc_id}
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["title"], "Q3 Growth Chat")
        self.assertEqual(data["document_id"], self.doc_id)

    def test_09_notes_crud(self):
        """
        Verifies CRUD operations for notes.
        """
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # 1. Create a note
        response = self.client.post(
            f"{settings.API_V1_STR}/notes/",
            headers=headers,
            json={"title": "Test Note", "content": "This is the content of the note."}
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["title"], "Test Note")
        self.assertEqual(data["content"], "This is the content of the note.")
        note_id = data["id"]
        
        # 2. List notes
        response = self.client.get(
            f"{settings.API_V1_STR}/notes/",
            headers=headers
        )
        self.assertEqual(response.status_code, 200)
        notes = response.json()
        self.assertTrue(any(n["id"] == note_id for n in notes))
        
        # 3. Update note
        response = self.client.put(
            f"{settings.API_V1_STR}/notes/{note_id}",
            headers=headers,
            json={"title": "Updated Test Note", "content": "Updated content."}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Updated Test Note")
        self.assertEqual(response.json()["content"], "Updated content.")
        
        # 4. Delete note
        response = self.client.delete(
            f"{settings.API_V1_STR}/notes/{note_id}",
            headers=headers
        )
        self.assertEqual(response.status_code, 204)
        
        # Verify note is deleted
        response = self.client.get(
            f"{settings.API_V1_STR}/notes/",
            headers=headers
        )
        notes = response.json()
        self.assertFalse(any(n["id"] == note_id for n in notes))

    @patch("app.services.llm.llm_service.retrieve_context")
    @patch("app.services.llm.llm_service.generate_openai_stream")
    def test_10_generate_guide(self, mock_openai_stream, mock_retrieve_context):
        """
        Verifies guide generator endpoint streams response tokens.
        """
        # Set up mocks
        mock_retrieve_context.return_value = []
        mock_openai_stream.return_value = ["token1", "token2"]
        
        # Create a document chunk for testing
        from app.models import DocumentChunk
        mock_chunk = DocumentChunk(
            id="mock-chunk-id",
            document_id=self.doc_id,
            page_number=1,
            content="Mock content for guide generation"
        )
        mock_retrieve_context.return_value = [mock_chunk]

        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Call endpoint
        response = self.client.post(
            f"{settings.API_V1_STR}/chats/generate-guide",
            headers=headers,
            json={
                "document_ids": [self.doc_id],
                "guide_type": "briefing"
            }
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        
        # Read the stream content
        content = response.content.decode()
        self.assertIn("token1", content)
        self.assertIn("token2", content)

if __name__ == "__main__":
    unittest.main()
