from dotenv import load_dotenv

load_dotenv()

# Define the system prompt
SYSTEM_PROMPT = """
You are an expert weather forecaster, who speak in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find thier location.
"""

# Create tools
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny {city}"


@dataclass
class Context:
    """Custom runtime context schema."""

    user_id: str


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id

    return "Florida" if user_id == "1" else "SF"


# Configure model
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-5-20250929", temperature=0.5, timeout=10, max_tokens=1000
)


# Define response format
# We use a dataclass here, but Pydantic models are also suported
@dataclass
class ResponseFormat:
    """Response schema for the agent"""

    # A punny response(always required)
    punny_response: str
    # Any intersting information about the weather if available
    weawther_conditions: str | None = None


# Add memory
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# Create and run the agent
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather outside?"}]},
    config=config,
    context=Context(user_id="1"),
)

print(response["structured_response"])

# Note that we can continue the conversation using the same 'thread_id'
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "thank you!",
            },
        ],
    },
    config=config,
    context=Context(user_id="1"),
)

print(response["structured_response"])
