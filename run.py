"""Launch the Steel Sentinel GPU Real-Time RL Learning Simulator server."""

from __future__ import annotations

import argparse
from aiohttp import web

from server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Steel Sentinel Real-Time RL Learning Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP/WebSocket port (default: 8080)")
    args = parser.parse_args()

    print(f"Steel Sentinel RL Lab starting at http://{args.host}:{args.port}/")
    web.run_app(create_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()

