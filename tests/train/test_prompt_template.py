from train.prompt_template import format_chat, INSTRUCTION


def test_format_chat_has_user_and_assistant():
    rec = format_chat("dirty input", "clean output")
    # MLX-LM chat format: messages list with role/content
    assert "messages" in rec
    roles = [m["role"] for m in rec["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_format_chat_includes_instruction():
    rec = format_chat("dirty", "clean")
    user_msg = next(m for m in rec["messages"] if m["role"] == "user")
    assert INSTRUCTION in user_msg["content"]


def test_format_chat_includes_dirty_input():
    rec = format_chat("specific dirty content here", "x")
    user_msg = next(m for m in rec["messages"] if m["role"] == "user")
    assert "specific dirty content here" in user_msg["content"]


def test_format_chat_assistant_is_clean():
    rec = format_chat("x", "specific clean output")
    asst_msg = next(m for m in rec["messages"] if m["role"] == "assistant")
    assert asst_msg["content"] == "specific clean output"
