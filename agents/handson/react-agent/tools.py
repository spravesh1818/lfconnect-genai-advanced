from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def get_weather(location: str) -> str:
    """Get weather for a location (demo stub)."""
    return f"The weather in {location} is sunny and 22°C."

TOOLS = [multiply, get_weather]