from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client

langfuse = get_client()

with langfuse.start_as_current_observation(as_type="span", name="ragu-smoke-test") as span:
    span.update(input={"question": "hello world!"}, output={"status": "ok"})

langfuse.flush()
print("Trace sent. Check Langfuse UI.")