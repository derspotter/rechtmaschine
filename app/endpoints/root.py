from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from shared import TEMPLATES_DIR

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML interface"""
    index_path = TEMPLATES_DIR / "index.html"
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
