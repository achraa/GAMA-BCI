"""main.py — P300 Hybrid BCI API (EEGNet + Eye Tracking)"""
import logging, os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
logger = logging.getLogger("p300_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from db.database import init_db
    init_db()
    logger.info("P300 Hybrid BCI API started")
    yield

app = FastAPI(title="P300 Hybrid BCI API",
              description="EEGNet (5-ch) + Eye Tracking probabilistik. Dataset: bigP3BCI SE001.",
              version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

from routes.upload   import router as upload_router
from routes.train    import router as train_router
from routes.models   import router as models_router
from routes.speller  import router as speller_router
from routes.evaluate import router as evaluate_router

app.include_router(upload_router,   prefix="/upload",   tags=["Upload"])
app.include_router(train_router,    prefix="/train",    tags=["Training"])
app.include_router(models_router,   prefix="/models",   tags=["Models"])
app.include_router(speller_router,  prefix="/speller",  tags=["Speller"])
app.include_router(evaluate_router, prefix="/evaluate", tags=["Evaluate"])


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "P300 Hybrid BCI API", "version": "1.0.0"}

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exc(req, exc):
    logger.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
