import srt
import httpx
import os
import json
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma:2b")

async def translate_text(
    text: str,
    context: str,
    previous_lines: list[str],
    model_name: str = MODEL_NAME,
    temperature: float = 0.3,
    system_prompt: str = ""
) -> str:
    """
    Translates text using the local Ollama instance with the specified context.
    """
    prompt = ""
    if system_prompt:
        prompt += f"{system_prompt}\n\n"
    else:
        prompt += "Translate the following English subtitle text to Czech.\n\n"

    if context:
        prompt += f"Context: {context}\n\n"

    if previous_lines:
        prompt += f"Previous lines for flow/context:\n"
        for line in previous_lines:
            prompt += f"- {line}\n"
        prompt += "\n"

    prompt += f"Text to translate:\n{text}\n\nTranslation:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
    except Exception as e:
        logger.error(f"Error during translation: {e}")
        return text # Fallback to original text if translation fails

async def process_translation(
    task_id: str,
    file_path: str,
    context: str,
    tasks_dict: dict,
    model_name: str = MODEL_NAME,
    temperature: float = 0.3,
    system_prompt: str = "",
    on_update=None
):
    """
    Background task to parse the SRT, translate line by line, and save the output.
    """
    tasks_dict[task_id]["status"] = "processing"
    if on_update:
        on_update()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            srt_content = f.read()

        subs = list(srt.parse(srt_content))
        tasks_dict[task_id]["total"] = len(subs)
        if on_update:
            on_update()

        translated_subs = []
        previous_lines = []

        for i, sub in enumerate(subs):
            original_text = sub.content.strip()

            # Simple chunking: keep last 3 lines for context
            context_lines = previous_lines[-3:]

            if original_text:
                translated_text = await translate_text(
                    original_text,
                    context,
                    context_lines,
                    model_name=model_name,
                    temperature=temperature,
                    system_prompt=system_prompt
                )
            else:
                translated_text = ""

            translated_sub = srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=translated_text
            )
            translated_subs.append(translated_sub)

            if original_text:
                previous_lines.append(f"{original_text} -> {translated_text}")

            tasks_dict[task_id]["progress"] = i + 1
            if on_update:
                on_update()

        # Write output file
        output_path = tasks_dict[task_id]["output_file"]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(translated_subs))

        tasks_dict[task_id]["status"] = "completed"
        if on_update:
            on_update()

    except Exception as e:
        logger.error(f"Failed to process task {task_id}: {e}")
        tasks_dict[task_id]["status"] = "failed"
        if on_update:
            on_update()

