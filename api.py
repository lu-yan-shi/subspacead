"""
SubspaceAD Anomaly Detection API — MeSquare platform service.

Usage:
    # Start server
    uvicorn api:app --host 0.0.0.0 --port 8703

    # Train
    curl -X POST http://localhost:8703/api/train \
      -F "files=@template.jpg" \
      -F "image_res=512"

    # Detect
    curl -X POST http://localhost:8703/api/detect \
      -F "file=@test.jpg" \
      -F "viz_mode=overlay"
"""

import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8703))
    print(f"Starting SubspaceAD v1.0.0")
    print(f"API docs: http://localhost:{port}/docs")
    uvicorn.run("app.main:app", host=host, port=port, reload=False, workers=1)
