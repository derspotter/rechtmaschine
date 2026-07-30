"""Collection-Allowlist des RAG-Proxys: nur kanzlei/jurisprudence/doktrin."""
import pytest
from fastapi import HTTPException

from endpoints.rag import validate_rag_collection


@pytest.mark.parametrize("ok", [None, "kanzlei", "jurisprudence", "doktrin"])
def test_known_collections_pass(ok):
    validate_rag_collection(ok)  # darf nicht raisen


def test_unknown_collection_rejected():
    with pytest.raises(HTTPException) as exc:
        validate_rag_collection("geheim")
    assert exc.value.status_code == 400
