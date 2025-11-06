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
    return FileResponse("/app/static/favicon.svg", media_type="image/svg+xml")
