"""aiohttp web server providing static assets, WebSocket live stream, and model persistence API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import WSMsgType, web

from .trainer import LearningManager

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
ALLOWED_DOWNLOADS = {"steel_sentinel_ppo.pt", "steel_sentinel_policy.onnx", "checkpoints.json"}


def create_app() -> web.Application:
    app = web.Application()
    manager = LearningManager(models_dir=MODELS_DIR, num_envs=64, rollout_steps=32)
    app["manager"] = manager

    async def on_startup(_app: web.Application) -> None:
        manager.start()

    async def on_cleanup(_app: web.Application) -> None:
        await manager.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    async def index_handler(_req: web.Request) -> web.FileResponse:
        return web.FileResponse(ROOT_DIR / "robot.html")

    async def status_handler(_req: web.Request) -> web.Response:
        return web.json_response(manager.get_init_payload())

    async def control_handler(req: web.Request) -> web.Response:
        body = await req.json()
        res = await manager.apply_control(body)
        return web.json_response(res)

    async def nudge_handler(_req: web.Request) -> web.Response:
        res = await manager.nudge()
        return web.json_response(res)

    async def save_handler(_req: web.Request) -> web.Response:
        meta = manager.save_now(export_onnx=True)
        await manager._broadcast(manager._stats_snapshot())
        return web.json_response({"ok": True, "model": meta})

    async def load_handler(_req: web.Request) -> web.Response:
        meta = await manager.load_saved()
        if meta is None:
            return web.json_response({"ok": False, "error": "No saved model checkpoint found"}, status=404)
        return web.json_response({"ok": True, "model": meta})

    async def reset_handler(_req: web.Request) -> web.Response:
        res = await manager.reset_training()
        return web.json_response({"ok": True, "state": res})

    async def download_handler(req: web.Request) -> web.StreamResponse:
        fname = req.match_info["filename"]
        if fname not in ALLOWED_DOWNLOADS:
            raise web.HTTPNotFound(text="Unknown model artifact")
        fpath = MODELS_DIR / fname
        if not fpath.exists():
            raise web.HTTPNotFound(text="Model file not saved yet")
        return web.FileResponse(
            fpath,
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    async def ws_handler(req: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(req)

        async def push(msg: dict) -> None:
            if not ws.closed:
                await ws.send_str(json.dumps(msg, separators=(",", ":")))

        manager.subscribe(push)
        try:
            await push(manager.get_init_payload())
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    if data.get("action") == "control":
                        await manager.apply_control(data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            manager.unsubscribe(push)
        return ws

    app.router.add_get("/", index_handler)
    app.router.add_get("/robot.html", index_handler)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/status", status_handler)
    app.router.add_post("/api/control", control_handler)
    app.router.add_post("/api/nudge", nudge_handler)
    app.router.add_post("/api/models/save", save_handler)
    app.router.add_post("/api/models/load", load_handler)
    app.router.add_post("/api/models/reset", reset_handler)
    app.router.add_get("/api/models/download/{filename}", download_handler)

    app.router.add_static("/styles", ROOT_DIR / "styles")
    app.router.add_static("/src", ROOT_DIR / "src")
    return app

