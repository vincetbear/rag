"""Serve the test UI."""

import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

# Resolve path relative to this file's actual location on disk
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@router.get("/", response_class=HTMLResponse)
async def ui():
    html_path = os.path.join(_STATIC_DIR, "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())
