from fastapi import FastAPI
from app.api.routes import router
from app.utils.logging import setup_logging

setup_logging()
app = FastAPI(title="LLM Document Analyzer")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)