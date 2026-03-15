#!/usr/bin/env python3
"""Wrapper to start uvicorn with asyncio loop (avoids uvloop PermissionError on macOS)."""
import sys
import os
import importlib.abc

# Force asyncio loop before uvicorn imports anything
os.environ["UVICORN_LOOP"] = "asyncio"

# Block uvloop from being imported (it causes PermissionError on macOS with system Python)
class UvloopBlocker(importlib.abc.MetaPathFinder):
    def find_module(self, fullname, path=None):
        if fullname == "uvloop":
            return self
    def load_module(self, fullname):
        raise ImportError("uvloop blocked by run_server.py to avoid macOS PermissionError")

sys.meta_path.insert(0, UvloopBlocker())

# Change to the backend directory so relative imports work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
    )
