from mcp.client.streamable_http import streamable_http_client
from strands import Agent
from strands.tools.mcp import MCPClient

def create_streamable_http_transport():
    return streamable_http_client("http://localhost:8000/mcp/")

mcp_client = MCPClient(create_streamable_http_transport)

agent = Agent(
    tools=[mcp_client]
)

print(agent("What is 81 divided by 9?"))