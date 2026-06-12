import srt
import httpx
import os
import json
import logging
import re

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "translategemma:4b")

DATA_DIR = os.getenv("DATA_DIR", "./data")
OLLAMA_LOG_FILE = os.path.join(DATA_DIR, "ollama.log")

def log_ollama(message: str):
    try:
        os.makedirs(os.path.dirname(OLLAMA_LOG_FILE), exist_ok=True)
        with open(OLLAMA_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        logger.error(f"Failed to write to Ollama log: {e}")

async def ensure_model_available(
    model_name: str,
    task_id: str = None,
    tasks_dict: dict = None,
    on_update = None
) -> bool:
    """
    Checks if a model is present locally in Ollama; if not, pulls it dynamically.
    """
    log_ollama(f"[CHECK] Verifying if model '{model_name}' is available at {OLLAMA_URL}...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            tags_response = await client.get(f"{OLLAMA_URL}/api/tags")
            tags_response.raise_for_status()
            tags_data = tags_response.json()
            models = []
            for m in tags_data.get("models", []):
                name = m.get("name", "")
                models.append(name)
                if ":" in name:
                    models.append(name.split(":")[0])

            if model_name in models or (model_name + ":latest") in models:
                log_ollama(f"[CHECK] Model '{model_name}' is already available locally.")
                return True

            log_ollama(f"[PULL] Model '{model_name}' not found locally. Initiating pull...")
            if tasks_dict and task_id:
                tasks_dict[task_id]["status"] = "pulling_model"
                if on_update:
                    on_update()

            pull_payload = {"name": model_name}
            async with client.stream('POST', f"{OLLAMA_URL}/api/pull", json=pull_payload, timeout=600.0) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            completed = data.get("completed", 0)
                            total = data.get("total", 0)
                            if total > 0:
                                percent = round((completed / total) * 100, 1)
                                log_ollama(f"[PULL STATUS] {status} - {percent}% ({completed}/{total})")
                            else:
                                log_ollama(f"[PULL STATUS] {status}")
                        except Exception:
                            log_ollama(f"[PULL STATUS] {line}")
            log_ollama(f"[PULL] Successfully finished pulling model '{model_name}'.")
            return True
    except Exception as e:
        log_ollama(f"[ERROR] Failed to verify or pull model '{model_name}': {e}")
        return False

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
    is_translategemma = "translategemma" in model_name.lower()
    text = text.replace("\n", " ").strip()

    if is_translategemma:
        lines_to_send = []
        for line in previous_lines:
            if " -> " in line:
                src = line.split(" -> ")[0]
                lines_to_send.append(src.replace("\n", " ").strip())
            else:
                lines_to_send.append(line.replace("\n", " ").strip())
        lines_to_send.append(text)
        prompt = "<<<source>>>en<<<target>>>cs<<<text>>>" + "\n".join(lines_to_send)
    else:
        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        else:
            prompt += "Translate the following English subtitle text to Czech. Produce only the translation without explanations.\n\n"

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

    log_ollama(f"[REQUEST] translate_text payload: {json.dumps(payload, ensure_ascii=False)}")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            resp_text = data.get("response", "").strip()
            log_ollama(f"[RESPONSE] translate_text result: '{resp_text}'")
            
            if is_translategemma:
                resp_lines = [l.strip() for l in resp_text.splitlines() if l.strip()]
                if resp_lines:
                    return resp_lines[-1]
            return resp_text
    except Exception as e:
        log_ollama(f"[ERROR] translate_text failed: {e}")
        logger.error(f"Error during translation: {e}")
        return text

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
    is_translategemma = "translategemma" in model_name.lower()
    cleaned_lines = [l.replace("\n", " ").strip() for l in lines]

    if is_translategemma:
        lines_to_send = []
        for line in previous_lines:
            if " -> " in line:
                src = line.split(" -> ")[0]
                lines_to_send.append(src.replace("\n", " ").strip())
            else:
                lines_to_send.append(line.replace("\n", " ").strip())
        num_context_lines = len(lines_to_send)
        lines_to_send.extend(cleaned_lines)
        prompt = "<<<source>>>en<<<target>>>cs<<<text>>>" + "\n".join(lines_to_send)
    else:
        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        else:
            prompt += "Translate the following English subtitle lines to Czech. Produce only the translation without explanations.\n\n"

        if context:
            prompt += f"Context/Topic: {context}\n\n"

        if previous_lines:
            prompt += f"Previous context for flow (do not translate these):\n"
            for line in previous_lines:
                prompt += f"- {line}\n"
            prompt += "\n"

        prompt += "Lines to translate:\n"
        for idx, line in enumerate(cleaned_lines, start=1):
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

    log_ollama(f"[REQUEST] translate_batch payload: {json.dumps(payload, ensure_ascii=False)}")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            raw_response = data.get("response", "").strip()
            log_ollama(f"[RESPONSE] translate_batch raw response: '{raw_response}'")
            
            if is_translategemma:
                resp_lines = [l.strip() for l in raw_response.splitlines() if l.strip()]
                translated = resp_lines[num_context_lines:]
                log_ollama(f"[PARSED translategemma] translations: {translated}")
                return translated
            else:
                raw_response = "1. " + raw_response
                translations = []
                for line in raw_response.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r"^\d+[\.\)]\s*(.*)$", line)
                    if match:
                        translations.append(match.group(1).strip())
                    elif translations:
                        translations[-1] += "\n" + line
                log_ollama(f"[PARSED] translations: {translations}")
                return translations
    except Exception as e:
        log_ollama(f"[ERROR] translate_batch failed: {e}")
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
        model_available = await ensure_model_available(
            model_name=model_name,
            task_id=task_id,
            tasks_dict=tasks_dict,
            on_update=on_update
        )
        
        if not model_available:
            logger.error(f"Model {model_name} is not available and could not be pulled.")
            tasks_dict[task_id]["status"] = "failed"
            if on_update:
                on_update()
            return

        tasks_dict[task_id]["status"] = "processing"
        if on_update:
            on_update()

        with open(file_path, "r", encoding="utf-8") as f:
            srt_content = f.read()

        subs = list(srt.parse(srt_content))
        tasks_dict[task_id]["total"] = len(subs)
        if on_update:
            on_update()

        translated_subs = []
        previous_lines = []

        i = 0
        while i < len(subs):
            batch = subs[i : i + batch_size]
            batch_texts = [sub.content.strip() for sub in batch]
            
            current_display_text = ", ".join([f'"{t}"' for t in batch_texts if t])
            tasks_dict[task_id]["current_text"] = current_display_text
            if on_update:
                on_update()

            non_empty_indices = [idx for idx, txt in enumerate(batch_texts) if txt]
            non_empty_texts = [batch_texts[idx] for idx in non_empty_indices]

            batch_translations = []
            if non_empty_texts:
                batch_translations = await translate_batch(
                    lines=non_empty_texts,
                    context=context,
                    previous_lines=previous_lines[-3:],
                    model_name=model_name,
                    temperature=temperature,
                    system_prompt=system_prompt
                )

            if len(batch_translations) != len(non_empty_texts):
                logger.warning(
                    f"Batch translation returned {len(batch_translations)} translations for "
                    f"{len(non_empty_texts)} lines. Falling back to line-by-line translation."
                )
                batch_translations = []
                for txt in non_empty_texts:
                    translated = await translate_text(
                        text=txt,
                        context=context,
                        previous_lines=previous_lines[-3:],
                        model_name=model_name,
                        temperature=temperature,
                        system_prompt=system_prompt
                    )
                    batch_translations.append(translated)

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

            previous_lines = previous_lines[-3:]

            i += len(batch)
            tasks_dict[task_id]["progress"] = min(i, len(subs))
            if on_update:
                on_update()

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
