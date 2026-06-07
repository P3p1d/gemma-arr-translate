import uuid
import os
import asyncio
from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
from app.translator import process_translation

import httpx
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma:2b")

@app.on_event("startup")
async def startup_event():
    # Attempt to pull the model if it's not present
    logger.info(f"Checking if model {MODEL_NAME} is available in Ollama at {OLLAMA_URL}...")
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Check if model exists
            tags_response = await client.get(f"{OLLAMA_URL}/api/tags")
            tags_response.raise_for_status()
            tags_data = tags_response.json()
            models = [m.get("name") for m in tags_data.get("models", [])]

            if MODEL_NAME not in models:
                logger.info(f"Model {MODEL_NAME} not found. Pulling now... This might take a while.")
                pull_payload = {"name": MODEL_NAME}
                # Streaming the pull request to avoid read timeouts on large models
                async with client.stream('POST', f"{OLLAMA_URL}/api/pull", json=pull_payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line:
                            # Log pull progress if necessary, or just consume
                            pass
                logger.info(f"Successfully pulled {MODEL_NAME}")
            else:
                logger.info(f"Model {MODEL_NAME} is already available.")
    except Exception as e:
        logger.error(f"Failed to check/pull model {MODEL_NAME}: {e}")

templates = Jinja2Templates(directory="app/templates")

# In-memory storage for task tracking (since persistence is not required right now)
# Structure: { task_id: {"status": "pending"|"processing"|"completed"|"failed", "filename": str, "progress": int, "total": int, "output_file": str} }
tasks = {}

os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/outputs", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    context: str = Form("")
):
    task_ids = []
    for file in files:
        task_id = str(uuid.uuid4())
        file_location = f"/tmp/uploads/{task_id}_{file.filename}"
        content = await file.read()
        with open(file_location, "wb") as file_object:
            file_object.write(content)

        tasks[task_id] = {
            "status": "pending",
            "filename": file.filename,
            "progress": 0,
            "total": 0,
            "output_file": f"/tmp/outputs/{task_id}_{file.filename}"
        }
        task_ids.append(task_id)

        background_tasks.add_task(process_translation, task_id, file_location, context, tasks)

    return {"task_ids": task_ids}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return task

@app.get("/download/{task_id}")
async def download_file(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "completed":
        return JSONResponse(status_code=404, content={"error": "File not found or not completed"})

    return FileResponse(path=task["output_file"], filename=task["filename"], media_type="text/plain")


from pydantic import BaseModel
from app.translator import translate_text

class LibreTranslateRequest(BaseModel):
    q: str | List[str]
    source: str = "en"
    target: str = "cs"
    format: str = "text"
    api_key: str = ""

@app.post("/translate")
async def libretranslate_emulation(request: LibreTranslateRequest):
    """
    Emulates the LibreTranslate POST /translate API for Bazarr integration.
    """
    texts = request.q if isinstance(request.q, list) else [request.q]
    translated_texts = []

    for text in texts:
        # Bazarr usually sends one line or a small chunk at a time.
        # We don't have rolling context here since it's stateless, but it will work for basic integration.
        translated = await translate_text(text, context="", previous_lines=[])
        translated_texts.append(translated)

    if isinstance(request.q, list):
        return {"translatedText": translated_texts}
    else:
        return {"translatedText": translated_texts[0]}

@app.get("/languages")
async def libretranslate_languages():
    """
    Emulates the LibreTranslate GET /languages API.
    """
    return [
        {"code": "en", "name": "English", "targets": ["cs"]},
        {"code": "cs", "name": "Czech", "targets": ["en"]}
    ]
