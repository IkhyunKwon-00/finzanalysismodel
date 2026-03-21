"""CLI: start the inference API server."""

import argparse
import uvicorn

from src.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Finz Analysis Model API")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    host = args.host or cfg["api"]["host"]
    port = args.port or cfg["api"]["port"]

    uvicorn.run("src.api.server:app", host=host, port=port, reload=False)
