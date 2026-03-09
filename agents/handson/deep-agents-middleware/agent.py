from typing import Callable
from dotenv import load_dotenv

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage

load_dotenv()


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


class TeacherStyleMiddleware(AgentMiddleware):
    """Inject a teaching-style response rule into every model call."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        addendum = (
            "\n\n## Middleware Instruction\n"
            "Answer in a teacher-friendly style.\n"
            "Use simple wording.\n"
            "When useful, include one short example.\n"
            "Keep the answer under 120 words."
        )

        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)

        return handler(modified_request)


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_deep_agent(
    model=model,
    tools=[multiply],
    middleware=[TeacherStyleMiddleware()],
    system_prompt="You are a helpful assistant."
)