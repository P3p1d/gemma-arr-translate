import os
from dotenv import load_dotenv
load_dotenv()
import modal
from modal import Image, App, Secret, method
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = App("bazarr-translator-gemma")

MODEL_DIR = "/model"
MODEL_NAME = "google/translategemma-12b-it"

# Define the vLLM image using the official CUDA devel image to provide nvcc for flashinfer JIT
vllm_image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "vllm>=0.6.0",
        "huggingface_hub",
        "hf-transfer",
        "bitsandbytes",
        "scipy",
        "fastapi[standard]"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1"
    })
)

# Authentication token for the webhook
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    # Read API_TOKEN from environment; fallback for convenience
    expected_token = os.environ.get("API_TOKEN", "bazarr_brains_secret")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials.credentials

@app.cls(
    image=vllm_image,
    gpu="l4",  # L4 has 24GB VRAM and costs ~$0.80/hr
    secrets=[
        Secret.from_dict({"HF_TOKEN": os.environ.get("HUGGING_FACE", "")}),
        Secret.from_dict({"API_TOKEN": os.environ.get("API_TOKEN", "bazarr_brains_secret")})
    ],
    timeout=600,
    scaledown_window=300,
    max_containers=1
)
class Translator:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        
        # Initialize vLLM engine with BitsAndBytes 8-bit quantization to fit in 24GB
        print(f"Loading model {MODEL_NAME} in 8-bit mode...")
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1, # 1 GPU
            max_model_len=4096, 
            enforce_eager=True, # Best for small memory constraints
            quantization="bitsandbytes",
            load_format="bitsandbytes"
        )
        print("Model loaded successfully!")

    @modal.fastapi_endpoint(method="POST")
    def translate(self, request: dict, token: str = Depends(verify_token)):
        """
        Expects a JSON body with:
        {
            "lines": ["line 1", "line 2", "line 3"],
            "source_lang": "en",
            "target_lang": "cs"
        }
        """
        from vllm import SamplingParams
        
        lines = request.get("lines", [])
        target_lang = request.get("target_lang", "cs")
        
        if not lines:
            return {"translated_lines": []}
            
        # Construct the prompt using XML tags which LLMs are highly trained to preserve
        text_to_translate = ""
        for i, line in enumerate(lines):
            text_to_translate += f"<line id=\"{i}\">{line}</line>\n"
            
        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=4096,
            stop=["\n\n\n", f"<line id=\"{len(lines)}\">"]
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text_to_translate
                    }
                ]
            }
        ]
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        
        # Parse the XML output safely
        import re
        translated_dict = {}
        # Find all <line id="X">...</line>
        matches = re.finditer(r'<line id="(\d+)">(.*?)</line>', generated_text, re.DOTALL)
        for match in matches:
            try:
                idx = int(match.group(1))
                text = match.group(2).strip()
                translated_dict[idx] = text
            except ValueError:
                pass
                
        # Reconstruct exactly len(lines) lines, falling back to original if the model skipped it
        translated_lines = []
        for i in range(len(lines)):
            translated_lines.append(translated_dict.get(i, lines[i]))
            
        if len(translated_dict) != len(lines):
            print(f"Warning: Model translated {len(translated_dict)} out of {len(lines)} lines. Missing lines fell back to English.")
            
        return {"translated_lines": translated_lines}
