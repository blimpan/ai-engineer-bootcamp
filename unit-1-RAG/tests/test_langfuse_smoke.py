import os

import pytest
from langfuse import get_client


@pytest.mark.integration
def test_langfuse_smoke_trace():
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        pytest.skip("Langfuse credentials are not configured")

    langfuse = get_client()

    with langfuse.start_as_current_observation(as_type="span", name="ragu-smoke-test") as span:
        span.update(input={"question": "hello world!"}, output={"status": "ok"})

    langfuse.flush()
