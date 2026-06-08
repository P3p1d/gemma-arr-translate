import os
import tempfile
import pytest
from unittest.mock import AsyncMock, patch

# Create a temporary directory for tests to avoid writing to host's ./data folder
temp_data_dir = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = temp_data_dir.name

# Prevent the startup event from executing full HTTP calls
with patch("app.main.startup_event", new_callable=AsyncMock) as mock_startup:
    from app.main import app, tasks, UPLOADS_DIR, OUTPUTS_DIR

from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Gemma Subtitle Translator" in response.text

def test_upload_files(mocker):
    # Mock process_translation background task so we don't actually run it
    mock_process = mocker.patch("app.main.process_translation", new_callable=AsyncMock)
    
    # Send a dummy srt file
    file_content = b"1\n00:00:01,000 --> 00:00:04,000\nHello"
    files = [("files", ("test.srt", file_content, "text/plain"))]
    
    upload_data = {
        "context": "some context",
        "model_name": "gemma:7b",
        "temperature": "0.5",
        "system_prompt": "Translate exactly."
    }
    
    response = client.post("/upload", files=files, data=upload_data)
    assert response.status_code == 200
    data = response.json()
    assert "task_ids" in data
    assert len(data["task_ids"]) == 1
    
    task_id = data["task_ids"][0]
    assert task_id in tasks
    assert tasks[task_id]["filename"] == "test.srt"
    assert tasks[task_id]["status"] == "pending"
    
    # Verify mock was called with correct parameters
    mock_process.assert_called_once()
    kwargs = mock_process.call_args[1]
    assert kwargs["model_name"] == "gemma:7b"
    assert kwargs["temperature"] == 0.5
    assert kwargs["system_prompt"] == "Translate exactly."
    
    # Clean up uploaded file if created
    uploaded_file = os.path.join(UPLOADS_DIR, f"{task_id}_test.srt")
    if os.path.exists(uploaded_file):
        os.remove(uploaded_file)

def test_get_all_tasks():
    task_id = "test-task-all"
    tasks[task_id] = {
        "status": "pending",
        "filename": "hello.srt",
        "progress": 0,
        "total": 0,
        "output_file": "output.srt"
    }
    
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert task_id in data
    assert data[task_id]["filename"] == "hello.srt"

def test_get_status():
    # Setup test task
    task_id = "test-task-123"
    tasks[task_id] = {
        "status": "processing",
        "filename": "movie.srt",
        "progress": 5,
        "total": 10,
        "output_file": "/tmp/outputs/test-task-123_movie.srt"
    }
    
    response = client.get(f"/status/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["filename"] == "movie.srt"
    
    # Test not found
    response_nf = client.get("/status/nonexistent")
    assert response_nf.status_code == 404

def test_download_file(tmp_path):
    task_id = "test-task-download"
    # Create a dummy output file
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(exist_ok=True)
    dummy_file = output_dir / "test-task-download_test.srt"
    dummy_file.write_text("1\n00:00:01,000 --> 00:00:04,000\nAhoj\n", encoding="utf-8")
    
    tasks[task_id] = {
        "status": "completed",
        "filename": "test.srt",
        "progress": 1,
        "total": 1,
        "output_file": str(dummy_file)
    }
    
    response = client.get(f"/download/{task_id}")
    assert response.status_code == 200
    assert response.text.replace("\r\n", "\n") == "1\n00:00:01,000 --> 00:00:04,000\nAhoj\n"
    
    # Test not found / not completed
    tasks["test-task-incomplete"] = {
        "status": "processing",
        "filename": "test.srt",
        "progress": 0,
        "total": 1,
        "output_file": str(dummy_file)
    }
    response_inc = client.get("/download/test-task-incomplete")
    assert response_inc.status_code == 404

@pytest.mark.asyncio
async def test_libretranslate_emulation(mocker):
    # Mock translate_text
    mocker.patch("app.main.translate_text", return_value="Ahoj")
    
    # Test single string
    payload_single = {"q": "Hello", "source": "en", "target": "cs"}
    response = client.post("/translate", json=payload_single)
    assert response.status_code == 200
    assert response.json() == {"translatedText": "Ahoj"}
    
    # Test array of strings
    payload_arr = {"q": ["Hello", "World"], "source": "en", "target": "cs"}
    response_arr = client.post("/translate", json=payload_arr)
    assert response_arr.status_code == 200
    assert response_arr.json() == {"translatedText": ["Ahoj", "Ahoj"]}

def test_libretranslate_languages():
    response = client.get("/languages")
    assert response.status_code == 200
    languages = response.json()
    assert len(languages) == 2
    assert languages[0]["code"] == "en"

# Cleanup temp data directory after tests complete
@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_dir():
    yield
    temp_data_dir.cleanup()
