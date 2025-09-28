from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Existing classic/XGB inference
from models.infer import load_deployed_model
# New: Keras text-only inference
from models.infer_keras_textonly import load_dl_model

# Retrieval
from retrieval.query import retrieve

# ------------- Logging setup -------------
logger = logging.getLogger("ips_api")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
logger.addHandler(_handler)

app = FastAPI(title="Intelligent Product Support API")

# Static demo page (served from service/static)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/demo", StaticFiles(directory=STATIC_DIR, html=True), name="demo")

# ------------- Lazy singletons -------------
_XGB_MODEL = None
_DL_MODEL = None

def _get_xgb_model():
    global _XGB_MODEL
    if _XGB_MODEL is None:
        logger.info("Loading classifier + featurizer...")
        _XGB_MODEL = load_deployed_model()
        logger.info("Loaded XGB model: %s", _XGB_MODEL.name)
    return _XGB_MODEL

def _get_dl_model():
    global _DL_MODEL
    if _DL_MODEL is None:
        logger.info("Loading Keras text-only model artifacts...")
        _DL_MODEL = load_dl_model()
        logger.info("Loaded DL model: %s", _DL_MODEL.name)
    return _DL_MODEL

# ------------- Helpers -------------

def _read_json_body(request: Request) -> Dict[str, Any]:
    try:
        body_bytes = request.scope.get("_body_bytes")
        if body_bytes is None:
            body_bytes = request._body  # type: ignore
        # Fallback to reading stream if not cached by ASGI
    except Exception:
        body_bytes = None

    # If not cached, read directly
    async def _read_stream():
        return await request.body()

    if not body_bytes:
        import anyio
        body_bytes = anyio.from_thread.run(_read_stream)

    try:
        return json.loads(body_bytes.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

def _tmp_ticket_json(payload: Dict[str, Any]) -> Path:
    """Write incoming ticket to a temp JSON (retrieval utils expect a file)."""
    tmp = Path("tmp_api_ticket.json")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return tmp

# ------------- Endpoints -------------

@app.on_event("startup")
def _print_urls():
    # Nice banner in the console
    base = os.environ.get("HOST_URL", "http://127.0.0.1:8000")
    print("===========================================")
    print(f" API running at:      {base}")
    print(f" Demo page available: {base}/demo/demo.html")
    print("===========================================")

@app.get("/")
def root():
    # Simple hint page
    return JSONResponse({"detail": "Endpoints: /classify, /classify_dl, /retrieve?k=N, /solve?k=N, /demo/demo.html"})

@app.get("/demo/demo.html")
def serve_demo():
    # Serve the demo page file directly (some environments need explicit route)
    html_path = STATIC_DIR / "demo.html"
    if not html_path.exists():
        raise HTTPException(404, "demo.html not found under service/static")
    return FileResponse(html_path)

@app.post("/classify")
def classify(request: Request):
    logger.info("[/classify] start")
    payload = _read_json_body(request)
    model = _get_xgb_model()
    out = model.predict(payload)
    logger.info("[/classify] done: %s (p=%.3f)", out.get("category"), out.get("confidence"))
    return JSONResponse(out)

@app.post("/classify_dl")
def classify_dl(request: Request):
    logger.info("[/classify_dl] start")
    payload = _read_json_body(request)
    model = _get_dl_model()
    out = model.predict(payload)
    logger.info("[/classify_dl] done: %s (p=%.3f)", out.get("category"), out.get("confidence"))
    return JSONResponse(out)

@app.post("/retrieve")
def retrieve_only(request: Request, k: int = 5):
    logger.info("[/retrieve] start (k=%d)", k)
    payload = _read_json_body(request)
    tmp_path = _tmp_ticket_json(payload)
    try:
        out = retrieve(tmp_path, k)
        logger.info("[/retrieve] done: cat=%s", out.get("predicted_category"))
        return JSONResponse(out)
    except Exception as e:
        logger.exception("Error in /retrieve")
        raise HTTPException(500, detail=str(e))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass

@app.post("/solve")
def solve(request: Request, k: int = 5):
    logger.info("[/solve] start (k=%d)", k)
    payload = _read_json_body(request)

    # 1) classify with XGB (kept as the default)
    model = _get_xgb_model()
    cls_out = model.predict(payload)

    # 2) retrieve top-K
    tmp_path = _tmp_ticket_json(payload)
    try:
        ret_out = retrieve(tmp_path, k)
    except Exception as e:
        logger.exception("Error in /solve")
        raise HTTPException(500, detail=str(e))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass

    results = ret_out.get("results", [])
    if results:
        top = results[0]
        summary = (
            f"Closest historical case: {top.get('doc_id')} "
            f"({top.get('category')}::{top.get('subcategory')}) "
            f"Resolution code: {top.get('resolution_code')} "
            f"Suggested next step: apply the known fix or verify the steps from the matched case; "
            f"if it diverges, re-run retrieval with broader filters."
        )
    else:
        summary = "No close matches found; try broadening filters or escalate."

    out = {
        "classification": cls_out,
        "retrieval": ret_out,
        "summary": summary,
    }
    logger.info("[/solve] done")
    return JSONResponse(out)
