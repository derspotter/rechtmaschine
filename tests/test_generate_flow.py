import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

try:
    import sqlalchemy
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("sqlalchemy must be installed in test environment") from exc

sqlalchemy.create_engine = MagicMock(return_value=MagicMock())

from fastapi.testclient import TestClient  # noqa: E402
from unittest.mock import patch  # noqa: E402

import app as app_module  # type: ignore  # noqa: E402
import database  # type: ignore  # noqa: E402

app = app_module.app
get_db = database.get_db


class FakeUploadResult:
    def __init__(self, file_id: str):
        self.id = file_id


class FakeFilesAPI:
    def upload(self, *args, **kwargs):
        return FakeUploadResult("file_dummy")


class FakeMessagesAPI:
    def __init__(self):
        self.last_model = None

    def create(self, *, model, **kwargs):
        self.last_model = model
        return type(
            "FakeResponse",
            (),
            {"content": [{"type": "text", "text": "Generated Klagebegründung"}]},
        )()


class FakeBetaAPI:
    def __init__(self):
        self.files = FakeFilesAPI()
        self.messages = FakeMessagesAPI()


class FakeAnthropicClient:
    def __init__(self):
        self.beta = FakeBetaAPI()


class DummyDB:
    def close(self):
        pass


def override_get_db():
    yield DummyDB()


app.dependency_overrides[get_db] = override_get_db


def fake_collect_selected_documents(selection, db):
    return {
        "anhoerung": [
            {"filename": "anhoerung1.pdf", "file_path": "/tmp/anhoerung1.pdf"},
        ],
        "bescheid": [
            {
                "filename": "bescheid_primary.pdf",
                "file_path": "/tmp/bescheid_primary.pdf",
                "role": "primary",
            }
        ],
        "rechtsprechung": [],
        "saved_sources": [
            {
                "id": "uuid-source-1",
                "title": "VG Beispiel",
                "url": "https://example.com",
                "document_type": "Rechtsprechung",
                "download_path": "/tmp/source.pdf",
            }
        ],
    }


def fake_upload_documents_to_claude(client, documents):
    return [
        {
            "type": "document",
            "source": {"type": "file", "file_id": "file_dummy"},
            "title": doc.get("filename") or doc.get("title"),
        }
        for doc in documents
    ]


def fake_verify_citations(text, selected_documents, sources_metadata=None):
    return {"cited": ["bescheid_primary.pdf"], "missing": [], "warnings": []}


client = TestClient(app)


def test_generate_structured_flow():
    payload = {
        "document_type": "Klagebegründung",
        "user_prompt": "Bitte erstelle eine Klagebegründung.",
        "selected_documents": {
            "anhoerung": ["anhoerung1.pdf"],
            "bescheid": {"primary": "bescheid_primary.pdf", "others": []},
            "rechtsprechung": [],
            "saved_sources": ["uuid-source-1"],
        },
    }

    with (
        patch("app.get_anthropic_client", return_value=FakeAnthropicClient()),
        patch("app._collect_selected_documents", side_effect=fake_collect_selected_documents),
        patch("app._upload_documents_to_claude", side_effect=fake_upload_documents_to_claude),
        patch("app.verify_citations", side_effect=fake_verify_citations),
    ):
        response = client.post("/generate", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["document_type"] == "Klagebegründung"
    assert "Generated Klagebegründung" in data["generated_text"]
    assert data["metadata"]["documents_used"]["bescheid"] == 1
    assert data["metadata"]["citations_found"] == 1
    assert data["metadata"]["missing_citations"] == []
    assert data["metadata"]["warnings"] == []


class FakeHTTPXResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text
        self.reason_phrase = "OK"

    def json(self):
        return self._json


class FakeHTTPXClient:
    def __init__(self, response=None):
        self.last_request = None
        self._response = response or FakeHTTPXResponse(status_code=200, json_data={"status": "ok"}, text="OK")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def put(self, url, auth=None, json=None):
        self.last_request = {"url": url, "auth": auth, "json": json}
        return self._response

    async def get(self, url, auth=None):
        self.last_request = {"url": url, "auth": auth}
        return self._response


def test_send_to_jlawyer_success():
    app_module.JLAWYER_BASE_URL = "https://jlawyer.example"
    app_module.JLAWYER_USERNAME = "user"
    app_module.JLAWYER_PASSWORD = "pass"
    app_module.JLAWYER_TEMPLATE_FOLDER_DEFAULT = "Klagebegruendungen"
    app_module.JLAWYER_PLACEHOLDER_KEY = "HAUPTTEXT"

    with patch("app.httpx.AsyncClient", lambda *args, **kwargs: FakeHTTPXClient()):
        response = client.post(
            "/send-to-jlawyer",
            json={
                "case_id": "AZ-2024",
                "template_name": "Vorlage.odt",
                "file_name": "output.odt",
                "generated_text": "Lorem ipsum"
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "j-lawyer" in data["message"].lower()


def test_get_jlawyer_templates():
    app_module.JLAWYER_BASE_URL = "https://jlawyer.example"
    app_module.JLAWYER_USERNAME = "user"
    app_module.JLAWYER_PASSWORD = "pass"
    app_module.JLAWYER_TEMPLATE_FOLDER_DEFAULT = "Standard"
    app_module.JLAWYER_PLACEHOLDER_KEY = "HAUPTTEXT"

    fake_response = FakeHTTPXResponse(status_code=200, json_data=["TemplateA.odt", "TemplateB.odt"], text="")

    with patch("app.httpx.AsyncClient", lambda *args, **kwargs: FakeHTTPXClient(fake_response)):
        response = client.get("/jlawyer/templates")

    assert response.status_code == 200
    data = response.json()
    assert data["templates"] == ["TemplateA.odt", "TemplateB.odt"]
    assert data["folder"] == "Standard"
