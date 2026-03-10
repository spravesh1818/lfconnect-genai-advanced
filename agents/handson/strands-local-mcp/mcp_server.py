from mcp.server import FastMCP

mcp = FastMCP("Calculator Server")

@mcp.tool(description="Add two numbers together")
def add(x: int, y: int) -> int:
    """Add two numbers and return the result."""
    return x + y

@mcp.tool(description="Subtract y from x")
def subtract(x: int, y: int) -> int:
    """Subtract y from x and return the result."""
    return x - y

@mcp.tool(description="Multiply two numbers together")
def multiply(x: int, y: int) -> int:
    """Multiply two numbers and return the result."""
    return x * y

@mcp.tool(description="Divide x by y")
def divide(x: int, y: int) -> float:
    """Divide x by y and return the result."""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

if __name__ == "__main__":
    mcp.run(transport="streamable-http")