import pytest
import os
import srt
from app.translator import translate_text, process_translation

@pytest.mark.asyncio
async def test_translate_text_success(mocker):
    # Mock httpx.AsyncClient.post
    mock_response = mocker.Mock()
    mock_response.raise_for_status = mocker.Mock()
    mock_response.json = mocker.Mock(return_value={"response": "Ahoj světe"})
    
    mock_post = mocker.patch("httpx.AsyncClient.post", return_value=mock_response)
    
    res = await translate_text("Hello world", "Context message", ["Prev line"])
    assert res == "Ahoj světe"
    
    # Verify the prompt contents passed to post
    mock_post.assert_called_once()
    kwargs = mock_post.call_args[1]
    assert "json" in kwargs
    prompt = kwargs["json"]["prompt"]
    assert "Hello world" in prompt
    assert "Context message" in prompt
    assert "Prev line" in prompt

@pytest.mark.asyncio
async def test_translate_text_failure(mocker):
    # Mock httpx.AsyncClient.post to raise an error
    mocker.patch("httpx.AsyncClient.post", side_effect=Exception("Connection refused"))
    
    res = await translate_text("Hello world", "", [])
    # Should fallback to original text
    assert res == "Hello world"

@pytest.mark.asyncio
async def test_process_translation(tmp_path, mocker):
    # Mock translate_text
    mocker.patch("app.translator.translate_text", side_effect=lambda text, ctx, prev, **kwargs: f"CS_{text}")
    
    # Create a dummy SRT file
    input_srt = tmp_path / "input.srt"
    srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello

2
00:00:05,000 --> 00:00:08,000
World
"""
    input_srt.write_text(srt_content, encoding="utf-8")
    
    output_srt = tmp_path / "output.srt"
    
    tasks_dict = {
        "test-task-1": {
            "status": "pending",
            "filename": "input.srt",
            "progress": 0,
            "total": 0,
            "output_file": str(output_srt)
        }
    }
    
    await process_translation("test-task-1", str(input_srt), "Test context", tasks_dict)
    
    assert tasks_dict["test-task-1"]["status"] == "completed"
    assert tasks_dict["test-task-1"]["progress"] == 2
    assert tasks_dict["test-task-1"]["total"] == 2
    
    # Read output
    output_content = output_srt.read_text(encoding="utf-8")
    parsed_output = list(srt.parse(output_content))
    assert len(parsed_output) == 2
    assert parsed_output[0].content == "CS_Hello"
    assert parsed_output[1].content == "CS_World"
