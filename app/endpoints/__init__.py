from fastapi import APIRouter

# Container for endpoint routers; individual modules should append their routers here.
routers: list[APIRouter] = []

__all__ = ["routers"]
