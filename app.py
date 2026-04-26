"""
Entry point for Hugging Face Spaces.
This file is required by Spaces to run your Docker container.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import your FastAPI app
from inference.api_server import app
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )