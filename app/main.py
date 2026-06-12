import os
import json
import logging
import asyncio
import httpx
import srt
from typing import List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Bazarr Subtitle Translator Brains")

MODAL_WORKSPACE = os.getenv("MODAL_WORKSPACE", "")
# In a real Modal deployment, you get an endpoint URL like:
# https://<workspace>--bazarr-translator-gemma-translator-translate.modal.run
# If the user doesn't know it yet, they can configure it in .env after `modal deploy`
MODAL_URL = os.getenv("MODAL_URL", "")
API_TOKEN = os.getenv("API_TOKEN", "bazarr_brains_secret")

CHUNK_SIZE = 150 # Number of subtitle lines to send to the LLM at once

class TranslateSrtRequest(BaseModel):
    srt_content: str
    target_lang: str = "cs"

async def translate_chunk(lines: List[str], target_lang: str) -> List[str]:
    if not MODAL_URL:
        logger.warning("MODAL_URL is not set. Returning original lines.")
        return lines
        
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "lines": lines,
        "target_lang": target_lang
    }
    
    try:
        logger.info(f"Sending {len(lines)} lines to Modal... First line: {lines[0][:50]}...")
        # Save intermediary request payload to /tmp for debugging
        with open("/tmp/last_request.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client: # Generous 10-minute timeout and follow 303 redirects from Modal
            response = await client.post(MODAL_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            translated = data.get("translated_lines", [])
            
            # Save intermediary response to /tmp for debugging
            with open("/tmp/last_response.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Received {len(translated)} translated lines. First line translated: {translated[0][:50]}...")
            
            # Fallback if the length doesn't match
            if len(translated) != len(lines):
                logger.error(f"Length mismatch: Sent {len(lines)} lines, got {len(translated)} lines.")
                # We could try to align them, but for now we fallback to returning original lines to avoid breaking SRT format entirely
                return lines
            return translated
    except Exception as e:
        logger.error(f"Error calling Modal endpoint: {e}")
        return lines


@app.post("/translate_srt")
async def translate_srt_endpoint(request: TranslateSrtRequest):
    """
    Accepts full SRT content, chunks it, calls Modal inference, and returns translated SRT content.
    """
    srt_content = request.srt_content
    target_lang = request.target_lang
    
    try:
        subs = list(srt.parse(srt_content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse SRT: {e}")
        
    if not subs:
        return {"translated_srt": srt_content}

    # Extract text lines
    lines = [sub.content for sub in subs]
    
    translated_lines = []
    
    # Process in chunks
    for i in range(0, len(lines), CHUNK_SIZE):
        chunk = lines[i:i + CHUNK_SIZE]
        logger.info(f"Translating chunk {i // CHUNK_SIZE + 1}/{(len(lines) + CHUNK_SIZE - 1) // CHUNK_SIZE}...")
        
        # Replace newlines within a single subtitle block with a special token so the LLM treats it as one line
        # But actually Gemma is smart enough to handle newlines if we format it right.
        # To be safe and ensure 1-to-1 mapping, we can flatten inner newlines during translation and restore them later.
        flattened_chunk = [line.replace("\n", " \\n ") for line in chunk]
        
        translated_chunk = await translate_chunk(flattened_chunk, target_lang)
        
        # Restore inner newlines
        restored_chunk = [line.replace(" \\n ", "\n") for line in translated_chunk]
        translated_lines.extend(restored_chunk)
        
    # Reconstruct SRT
    for i, sub in enumerate(subs):
        if i < len(translated_lines):
            sub.content = translated_lines[i]
            
    translated_srt = srt.compose(subs)
    
    return {"translated_srt": translated_srt}

@app.get("/")
async def health_check():
    return {"status": "ok", "modal_url_configured": bool(MODAL_URL)}
