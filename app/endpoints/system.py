from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, FileResponse

router = APIRouter()


@router.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse(content={"status": "healthy"}, status_code=200)


@router.get("/favicon.ico")
async def favicon():
    """Serve favicon (prevents 404 errors)"""
    favicon_path = Path("/app/static/favicon.svg")
    if not favicon_path.exists():
        return JSONResponse(content={}, status_code=204)
    return FileResponse(str(favicon_path), media_type="image/svg+xml")
