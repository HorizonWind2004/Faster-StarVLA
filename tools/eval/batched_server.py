"""starVLA policy server that batches concurrent inference requests.

    python tools/eval/batched_server.py --ckpt_path CKPT --port PORT --use_bf16 [--max_batch 64]

Same protocol, model loading and response format as starVLA's
``deployment/model_server/server_policy.py``; only the dispatch differs. starVLA's server runs
``predict_action`` inline in the event loop, one request at a time, so with many simulation
clients per GPU they queue behind each other. Here every connection puts its infer request on a
queue; one worker takes everything queued (up to ``max_batch``), groups requests whose non-example
arguments are equal, concatenates their ``examples`` into one ``predict_action`` call on a worker
thread (the event loop keeps receiving meanwhile) and splits the ``[B, T, D]`` actions back.
Requests that arrive while the GPU is busy form the next batch, so no waiting window is needed.
Batched and single-request inference can differ in the last bits (padding of the prompts in a
batch, bf16); success rates are compared against the unbatched server in the speed test.
"""

import argparse
import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import websockets.asyncio.server
import websockets.frames

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools import msgpack_numpy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _group_key(payload: dict) -> str:
    return repr(sorted((k, repr(v)) for k, v in payload.items() if k != "examples"))


class BatchedPolicyServer(WebsocketPolicyServer):

    def __init__(self, *args, max_batch: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_batch = max_batch
        self._pool = ThreadPoolExecutor(max_workers=1)  # one GPU, one inference at a time
        self._stats = {"batches": 0, "requests": 0, "t0": time.time()}

    async def run(self):
        self._queue = asyncio.Queue()
        worker = asyncio.create_task(self._batch_loop())
        try:
            # No keepalive pings: a simulation client blocks in synchronous code while it builds a
            # heavy LIBERO-Plus scene and cannot answer within websockets' 20 s ping timeout, so the
            # server dropped such clients (3 of 96 workers lost their ~100 tasks in one run).
            async with websockets.asyncio.server.serve(self._handler, self._host, self._port, compression=None,
                                                       max_size=None, ping_interval=None) as server:
                await server.serve_forever()
        finally:
            worker.cancel()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        loop = asyncio.get_running_loop()
        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()
                mtype = msg.get("type", "infer")
                payload = msg.get("payload", msg)
                if mtype in ("infer", "predict_action") and isinstance(payload, dict) and "examples" in payload:
                    fut = loop.create_future()
                    await self._queue.put((msg.get("request_id", "default"), payload, fut))
                    ret = await fut
                else:
                    ret = self._route_message(msg)
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Internal server error.")
                raise

    def _infer(self, examples, kwargs):
        return self._policy.predict_action(examples=examples, **kwargs)

    async def _batch_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self._queue.get()]
            while len(batch) < self._max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            groups = {}
            for item in batch:
                groups.setdefault(_group_key(item[1]), []).append(item)
            for items in groups.values():
                examples, counts = [], []
                for _, payload, _ in items:
                    ex = list(payload["examples"])
                    examples += ex
                    counts.append(len(ex))
                kwargs = {k: v for k, v in items[0][1].items() if k != "examples"}
                try:
                    out = await loop.run_in_executor(self._pool, self._infer, examples, kwargs)
                    actions = np.asarray(out["actions"])
                    start = 0
                    for (req_id, _, fut), n in zip(items, counts):
                        data = {k: (np.asarray(v)[start:start + n] if isinstance(v, np.ndarray) and len(v) == len(examples)
                                    else v) for k, v in out.items()}
                        data["actions"] = actions[start:start + n]
                        start += n
                        fut.set_result({"status": "ok", "ok": True, "type": "inference_result",
                                        "request_id": req_id, "data": data})
                except Exception as e:  # report to every waiting client, keep serving
                    logging.exception("batched inference failed")
                    for req_id, _, fut in items:
                        if not fut.done():
                            fut.set_result({"status": "error", "ok": False, "type": "inference_result",
                                            "request_id": req_id, "error": {"message": str(e)}})
            self._stats["batches"] += 1
            self._stats["requests"] += len(batch)
            if self._stats["batches"] % 500 == 0:
                s = self._stats
                logging.warning("[batched] %d batches, %d requests, mean batch %.1f, %.1f req/s", s["batches"],
                                s["requests"], s["requests"] / s["batches"], s["requests"] / (time.time() - s["t0"]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--port", type=int, default=10093)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--max_batch", type=int, default=64)
    p.add_argument("--idle_timeout", type=int, default=-1)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    wrapper = PolicyServerWrapper(ckpt_path=args.ckpt_path, device="cuda", use_bf16=args.use_bf16, config_overrides=[])
    server = BatchedPolicyServer(policy=wrapper, host="0.0.0.0", port=args.port, idle_timeout=args.idle_timeout,
                                 metadata=dict(wrapper.metadata), max_batch=args.max_batch)
    server.serve_forever()


if __name__ == "__main__":
    main()
