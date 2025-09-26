from __future__ import annotations

from .core import router as core_router
from .fetch import router as fetch_router
from .analytics import router as analytics_router
from .management import router as management_router
from fastapi import APIRouter

router = APIRouter()

# Include specialized router modules
router.include_router(core_router, prefix="", tags=["calendar"])
router.include_router(fetch_router, prefix="", tags=["calendar"]) 
router.include_router(analytics_router, prefix="", tags=["calendar"])
router.include_router(management_router, prefix="", tags=["calendar"])

