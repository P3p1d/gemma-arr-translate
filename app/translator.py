import srt
import httpx
import os
import json
import logging
import re

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
    Translates a single line using the local Ollama instance.
    """
    prompt = ""
    if system_prompt:
        prompt += f"{system_prompt}\n\n"
    else:
        prompt += "Translate the following English subtitle text to Czech.\n\n"

    if context:
        prompt += f"Context: {context}\n\n"

    if previous_lines:
        prompt += f"Previous context for flow:\n"
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

async def translate_batch(
    lines: list[str],
    context: str,
    previous_lines: list[str],
    model_name: str = MODEL_NAME,
    temperature: float = 0.3,
    system_prompt: str = ""
) -> list[str]:
    """
    Translates a list of lines in a single Ollama API call, carrying over previous lines context.
    """
    prompt = ""
    if system_prompt:
        prompt += f"{system_prompt}\n\n"
    else:
        prompt += "Translate the following English subtitle lines to Czech.\n\n"

    if context:
        prompt += f"Context/Topic: {context}\n\n"

    if previous_lines:
        prompt += f"Previous context for flow (do not translate these):\n"
        for line in previous_lines:
            prompt += f"- {line}\n"
        prompt += "\n"

    prompt += "Lines to translate:\n"
    for idx, line in enumerate(lines, start=1):
        prompt += f"{idx}. {line}\n"
    
    prompt += "\nCzech translations:\n1."

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
            raw_response = data.get("response", "").strip()
            
            # Reconstruct the "1." prefix since it was in the prompt suffix
            raw_response = "1. " + raw_response
            
            # Parse numbered list
            translations = []
            for line in raw_response.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Match "1. Translation text" or "1) Translation text" or "1.Translation text"
                match = re.match(r"^\d+[\.\)]\s*(.*)$", line)
                if match:
                    translations.append(match.group(1).strip())
                elif translations:
                    # Append multi-line translation to the last parsed translation
                    translations[-1] += "\n" + line

            return translations
    except Exception as e:
        logger.error(f"Error during batch translation: {e}")
        return []

async def process_translation(
    task_id: str,
    file_path: str,
    context: str,
    tasks_dict: dict,
    model_name: str = MODEL_NAME,
    temperature: float = 0.3,
    system_prompt: str = "",
    batch_size: int = 5,
    on_update=None
):
    """
    Background task to parse the SRT, translate in batches with inter-batch context, and save output.
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
        previous_lines = [] # Stores last few translated lines for context: "source -> translation"

        # Group subs into batches
        i = 0
        while i < len(subs):
            batch = subs[i : i + batch_size]
            batch_texts = [sub.content.strip() for sub in batch]
            
            # Update GUI progress detail
            current_display_text = ", ".join([f'"{t}"' for t in batch_texts if t])
            tasks_dict[task_id]["current_text"] = current_display_text
            if on_update:
                on_update()

            # Filter out empty lines for translation, but remember their positions
            non_empty_indices = [idx for idx, txt in enumerate(batch_texts) if txt]
            non_empty_texts = [batch_texts[idx] for idx in non_empty_indices]

            batch_translations = []
            if non_empty_texts:
                # Attempt batch translation
                batch_translations = await translate_batch(
                    lines=non_empty_texts,
                    context=context,
                    previous_lines=previous_lines[-3:],
                    model_name=model_name,
                    temperature=temperature,
                    system_prompt=system_prompt
                )

            # If batch translation fails or returns mismatch length, fall back to line-by-line
            if len(batch_translations) != len(non_empty_texts):
                logger.warning(
                    f"Batch translation returned {len(batch_translations)} translations for "
                    f"{len(non_empty_texts)} lines. Falling back to line-by-line translation."
                )
                batch_translations = []
                for txt in non_empty_texts:
                    # Translate single line carrying over the sliding history
                    translated = await translate_text(
                        text=txt,
                        context=context,
                        previous_lines=previous_lines[-3:],
                        model_name=model_name,
                        temperature=temperature,
                        system_prompt=system_prompt
                    )
                    batch_translations.append(translated)

            # Reconstruct full batch (including empty lines)
            trans_idx = 0
            for idx, txt in enumerate(batch_texts):
                sub = batch[idx]
                if txt:
                    translated_text = batch_translations[trans_idx]
                    trans_idx += 1
                    previous_lines.append(f"{txt} -> {translated_text}")
                else:
                    translated_text = ""

                translated_sub = srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=translated_text
                )
                translated_subs.append(translated_sub)

            # Keep only last 3 lines for context window to prevent prompt bloat
            previous_lines = previous_lines[-3:]

            i += len(batch)
            tasks_dict[task_id]["progress"] = min(i, len(subs))
            if on_update:
                on_update()

        # Write output file
        output_path = tasks_dict[task_id]["output_file"]
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(translated_subs))

        tasks_dict[task_id]["status"] = "completed"
        tasks_dict[task_id]["current_text"] = ""
        if on_update:
            on_update()

    except Exception as e:
        logger.error(f"Failed to process task {task_id}: {e}")
        tasks_dict[task_id]["status"] = "failed"
        if on_update:
            on_update()
