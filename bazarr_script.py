import sys
import os
import json
import urllib.request

# Configuration
HOMELAB_URL = "http://192.168.0.100:6969/translate_srt"
TARGET_LANG = "cs"

def main():
    if len(sys.argv) < 2:
        print("Usage: python bazarr_script.py <path_to_srt>")
        sys.exit(1)
        
    srt_path = sys.argv[1]
    
    if not os.path.exists(srt_path):
        print(f"File not found: {srt_path}")
        sys.exit(1)
        
    filename = os.path.basename(srt_path)
    
    # Check if the file is already the target language (e.g. .cs.srt)
    if filename.endswith(f".{TARGET_LANG}.srt"):
        print(f"Subtitle is already in target language ({TARGET_LANG}). Skipping translation.")
        sys.exit(0)
        
    print(f"Starting translation for: {srt_path}")
    
    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()
        
    payload = json.dumps({
        "srt_content": srt_content,
        "target_lang": TARGET_LANG
    }).encode("utf-8")
    
    req = urllib.request.Request(
        HOMELAB_URL, 
        data=payload, 
        headers={"Content-Type": "application/json"}
    )
    
    try:
        # Increased timeout to 3600 seconds (1 hour) to handle massive movie subtitles
        # and the initial 3-minute Modal cold boot without disconnecting prematurely.
        with urllib.request.urlopen(req, timeout=3600) as response:
            result = json.loads(response.read().decode("utf-8"))
            translated_srt = result.get("translated_srt", "")
            
            if not translated_srt:
                print("Translation returned empty result.")
                sys.exit(1)
                
            # Determine new filename (replace existing lang code with target lang code)
            # Example: movie.en.srt -> movie.cs.srt
            
            abs_srt_path = os.path.abspath(srt_path)
            base_without_ext = abs_srt_path[:-4] # remove .srt
            
            if "." in os.path.basename(base_without_ext):
                # Replace the last part (the language code)
                parts = base_without_ext.rsplit(".", 1)
                new_path = f"{parts[0]}.{TARGET_LANG}.srt"
            else:
                new_path = f"{base_without_ext}.{TARGET_LANG}.srt"
                
            # Write translated content to new file
            new_path_abs = os.path.abspath(new_path)
            with open(new_path_abs, "w", encoding="utf-8") as f:
                f.write(translated_srt)
                
            print(f"Successfully translated and saved to: {new_path_abs}")
            
            # Delete original file if we changed the filename
            # if new_path != srt_path:
            #     os.remove(srt_path)
            #     print(f"Deleted original file: {srt_path}")
                
    except Exception as e:
        print(f"Error during translation request: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
