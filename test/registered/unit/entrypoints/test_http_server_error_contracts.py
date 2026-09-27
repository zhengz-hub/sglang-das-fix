from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi.testclient import TestClient

from sglang.srt.entrypoints.http_server import _extract_validation_param, app


client = TestClient(app)


def _assert_openai_bad_request(response, param):
    assert response.status_code == 400
    assert set(response.json()) == {"error"}
    error = response.json()["error"]
    assert set(error) == {"object", "message", "type", "param", "code"}
    assert error["object"] == "error"
    assert error["type"] == "Bad Request"
    assert error["param"] == param
    assert error["code"] == 400


def test_extract_validation_param_uses_only_first_error_location():
    errors = [
        {
            "loc": ("body", "tools", 0, "function", "parameters"),
            "msg": "mentions a misleading field name",
            "ctx": {"server_context_length": 1},
            "input": {"unrelated": True},
            "type": "value_error",
        },
        {"loc": ("body", "ignored"), "msg": "ignored"},
    ]

    assert _extract_validation_param(errors) == "tools[0].function.parameters"
    assert _extract_validation_param([{"loc": ("body",)}]) is None
    assert _extract_validation_param([]) is None


def test_chat_scalar_pydantic_error_uses_location_param():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": "not-an-integer",
        },
    )

    _assert_openai_bad_request(response, "max_tokens")


def test_chat_nested_pydantic_error_uses_real_location_param():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"parameters": {"type": "object"}},
                }
            ],
        },
    )

    _assert_openai_bad_request(response, "tools[0].function.name")


def test_content_type_validation_reports_header_name():
    response = client.post("/v1/chat/completions", content=b"{}")

    _assert_openai_bad_request(response, "content-type")


def test_responses_pydantic_error_uses_same_envelope_and_field_types():
    response = client.post("/v1/responses", json={})

    _assert_openai_bad_request(response, "input")
    assert isinstance(response.json()["error"]["type"], str)


def test_anthropic_pydantic_error_keeps_anthropic_envelope():
    response = client.post("/v1/messages", json={})

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "object" not in body["error"]
    assert "param" not in body["error"]
