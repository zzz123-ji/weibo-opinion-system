# -*- coding: utf-8 -*-
"""10 单条推理服务：FastAPI（优先）/ 标准库 HTTP 回退 + 自检。

用法
----
    python src/10_inference_service.py --self-test
    python src/10_inference_service.py --serve --host 127.0.0.1 --port 8000

接口
----
    GET  /health
    POST /predict
         {"text": "...", "image_path": "data/crawl/images/xxx.jpg", "explain": true}
         或 {"text": "...", "image_base64": "data:image/jpeg;base64,..."}

边界
----
- 只支持文本 + 图像；不伪造视频/语音能力；
- 强度为概率映射演示值，不是人工 -3~3 标注；
- 服务默认只监听 127.0.0.1，不对外网开放。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import OUT, ROOT, write_json
from lib_infer import SentimentInference

HOST_DEFAULT = "127.0.0.1"
PORT_DEFAULT = 8000
_ENGINE: SentimentInference | None = None


def get_engine() -> SentimentInference:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SentimentInference()
    return _ENGINE


def predict_payload(payload: dict) -> dict:
    text = str(payload.get("text", "") or "")
    if not text.strip():
        raise ValueError("缺少 text")
    return get_engine().predict(
        text=text,
        image_path=payload.get("image_path") or None,
        image_base64=payload.get("image_base64") or None,
        explain=bool(payload.get("explain", True)),
    )


def run_self_test() -> int:
    t0 = time.time()
    eng = get_engine()
    sample = None
    csv_path = os.path.join(ROOT, "data", "crawl", "weibo_1000.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                img = row.get("image_file", "")
                if img and os.path.exists(os.path.join(ROOT, "data", "crawl", img.replace("/", os.sep))):
                    sample = row
                    break
    text = (sample or {}).get("text") or "武汉加油，感谢一线医护人员！"
    image_path = None
    if sample:
        image_path = os.path.join("data", "crawl", sample["image_file"])
    examples = []
    examples.append({"name": "text_only", "result": eng.predict(text, explain=True, explain_image=False)})
    if image_path:
        examples.append({"name": "text_image", "result": eng.predict(text, image_path=image_path, explain=True)})
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine_loaded_at": eng.loaded_at,
        "model_dir": eng.model_dir,
        "n_examples": len(examples),
        "examples": examples,
        "total_latency_ms": round((time.time() - t0) * 1000, 1),
        "note": "自检结果只证明推理链路可运行；强度为概率映射演示值，不是人工强度标注。",
    }
    write_json(os.path.join(OUT, "inference_self_test.json"), result)
    print("[OK] 推理自检完成")
    for ex in examples:
        r = ex["result"]
        print(f"  {ex['name']}: {r['prediction']['label_name']} "
              f"p={r['prediction']['probabilities']} "
              f"gate={r['modalities']['gate']} latency={r['latency_ms']}ms")
    print(f"  结果 -> {os.path.relpath(os.path.join(OUT, 'inference_self_test.json'), ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# FastAPI（可选）
# ---------------------------------------------------------------------------
def build_fastapi():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        text: str
        image_path: str | None = None
        image_base64: str | None = None
        explain: bool = True

    app = FastAPI(title="微博多模态情感单条推理", version="1.0")

    @app.get("/health")
    def health():
        eng = get_engine()
        return {"status": "ok", "loaded_at": eng.loaded_at,
                "modalities": ["text", "image"], "model_dir": eng.model_dir}

    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            return predict_payload(req.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return app


# ---------------------------------------------------------------------------
# 标准库 HTTP 回退（无需安装 FastAPI）
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    server_version = "WeiboSentimentHTTP/1.0"

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_GET(self):
        if self.path.split("?")[0] == "/health":
            eng = get_engine()
            self._send(200, {"status": "ok", "loaded_at": eng.loaded_at,
                             "modalities": ["text", "image"], "model_dir": eng.model_dir})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            self._send(200, predict_payload(payload))
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        return


def run_stdlib(host: str, port: int):
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"[serve] 标准库 HTTP 回退：http://{host}:{port}  (GET /health, POST /predict)")
    print("        提示：安装 fastapi/uvicorn 后可用同一命令启动 FastAPI 版本。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop]")
    finally:
        httpd.server_close()


def run_fastapi(host: str, port: int):
    import uvicorn
    app = build_fastapi()
    print(f"[serve] FastAPI：http://{host}:{port}  (docs: /docs)")
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    ap = argparse.ArgumentParser(description="微博多模态单条推理服务")
    ap.add_argument("--serve", action="store_true", help="启动 HTTP 服务（默认只跑自检）")
    ap.add_argument("--self-test", action="store_true", help="跑推理自检并写 output/inference_self_test.json")
    ap.add_argument("--host", default=HOST_DEFAULT)
    ap.add_argument("--port", type=int, default=PORT_DEFAULT)
    args = ap.parse_args()

    if args.self_test or not args.serve:
        raise SystemExit(run_self_test())

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        run_fastapi(args.host, args.port)
    except ImportError:
        print("[WARN] 未安装 fastapi/uvicorn，使用标准库 HTTP 回退。")
        run_stdlib(args.host, args.port)


if __name__ == "__main__":
    main()
