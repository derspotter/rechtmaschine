"""RagChat-ORM: to_dict-Formen für Liste (ohne Messages) und Detail."""
import uuid
from datetime import datetime

from models import RagChat


def _chat():
    chat = RagChat(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        title="Wie argumentieren wir § 60 Abs. 5?",
        collections=["kanzlei", "doktrin"],
        messages=[{"role": "user", "content": "F?", "created_at": "2026-07-30T12:00:00Z"}],
        created_at=datetime(2026, 7, 30, 12, 0, 0),
        updated_at=datetime(2026, 7, 30, 12, 5, 0),
    )
    return chat


def test_to_dict_list_form_omits_messages():
    d = _chat().to_dict()
    assert d["title"].startswith("Wie argumentieren")
    assert d["collections"] == ["kanzlei", "doktrin"]
    assert "messages" not in d
    assert d["message_count"] == 1
    assert d["updated_at"] == "2026-07-30T12:05:00"


def test_to_dict_detail_form_includes_messages():
    d = _chat().to_dict(include_messages=True)
    assert d["messages"][0]["content"] == "F?"
