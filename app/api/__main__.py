"""Entry point for running the API server directly: python -m app.api"""

import uvicorn

from app.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
