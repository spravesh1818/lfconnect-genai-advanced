import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, END

from src.state import TravelState
from src.tools import recommend_destination, search_flights, search_hotels

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=MODEL, temperature=0)

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a supervisor for a travel assistant.\n"
         "Your job: decide which specialist agent should run next.\n"
         "Rules:\n"
         "- If destination is missing -> destination\n"
         "- Else if flights missing -> flight\n"
         "- Else if hotels missing -> hotel\n"
         "- Else -> final\n"
         "Return ONLY one word: destination, flight, hotel, or final."
        ),
        ("human",
         "User request:\n{user_request}\n\n"
         "Current state:\n"
         "destination={destination}\n"
         "flights={flights}\n"
         "hotels={hotels}\n"
        ),
    ]
)

def supervisor_node(state: TravelState) -> TravelState:
    # last human request (or first; here we use latest user message)
    user_request = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_request = m.content
            break

    decision = llm.invoke(
        supervisor_prompt.format_messages(
            user_request=user_request,
            destination=state.get("destination"),
            flights=state.get("flights"),
            hotels=state.get("hotels"),
        )
    ).content.strip().lower()

    if decision not in {"destination", "flight", "hotel", "final"}:
        # fallback safety
        if not state.get("destination"):
            decision = "destination"
        elif not state.get("flights"):
            decision = "flight"
        elif not state.get("hotels"):
            decision = "hotel"
        else:
            decision = "final"

    state["next"] = decision  # type: ignore
    return state

def destination_node(state: TravelState) -> TravelState:
    # find latest user request
    user_request = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_request = m.content
            break

    dest = recommend_destination(user_request)
    state["destination"] = dest
    state["messages"].append(AIMessage(content=f"[DestinationAgent] Recommended: {dest}"))
    state["next"] = "supervisor"
    return state


def flight_node(state: TravelState) -> TravelState:
    destination = state.get("destination") or "the chosen destination"
    flights = search_flights(destination)
    state["flights"] = flights
    state["messages"].append(AIMessage(content=f"[FlightAgent] {flights}"))
    state["next"] = "supervisor"
    return state


def hotel_node(state: TravelState) -> TravelState:
    destination = state.get("destination") or "the chosen destination"
    hotels = search_hotels(destination)
    state["hotels"] = hotels
    state["messages"].append(AIMessage(content=f"[HotelAgent] {hotels}"))
    state["next"] = "supervisor"
    return state


final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful travel assistant. Create a clean final answer.\n"
         "Use the collected results. Be concise.\n"
        ),
        ("human",
         "User request:\n{user_request}\n\n"
         "Destination:\n{destination}\n\n"
         "Flights:\n{flights}\n\n"
         "Hotels:\n{hotels}\n"
        ),
    ]
)

def final_node(state: TravelState) -> TravelState:
    user_request = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_request = m.content
            break

    final_text = llm.invoke(
        final_prompt.format_messages(
            user_request=user_request,
            destination=state.get("destination"),
            flights=state.get("flights"),
            hotels=state.get("hotels"),
        )
    ).content

    state["messages"].append(AIMessage(content=final_text))
    return state


def route_from_supervisor(state: TravelState) -> str:
    return state.get("next") or "destination"

builder = StateGraph(TravelState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("destination", destination_node)
builder.add_node("flight", flight_node)
builder.add_node("hotel", hotel_node)
builder.add_node("final", final_node)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "destination": "destination",
        "flight": "flight",
        "hotel": "hotel",
        "final": "final",
    },
)

builder.add_edge("destination", "supervisor")
builder.add_edge("flight", "supervisor")
builder.add_edge("hotel", "supervisor")
builder.add_edge("final", END)

graph = builder.compile()