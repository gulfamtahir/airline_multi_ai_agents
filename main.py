import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
import psycopg

from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents import create_agent
from langgraph.types import Command
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (AnyMessage, HumanMessage, SystemMessage, AIMessage)
from langchain_groq import ChatGroq

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)


# ── Per-agent LLM usage tracking ─────────────────────────────────────────────
class AgentUsageTracker(BaseCallbackHandler):
    """Counts real LLM calls and tokens for whatever runs with this callback attached."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1

        usage = None
        try:
            message = response.generations[0][0].message
            usage = getattr(message, "usage_metadata", None)
        except (IndexError, AttributeError):
            pass

        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        else:
            token_usage = (response.llm_output or {}).get("token_usage", {})
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total_tokens

    def as_dict(self):
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def merge_usage(existing: dict, new: dict) -> dict:
    """Reducer: adds each agent's per-run stats onto its running total."""
    merged = dict(existing)
    for agent, stats in new.items():
        if agent in merged:
            merged[agent] = {key: merged[agent][key] + stats[key] for key in stats}
        else:
            merged[agent] = dict(stats)
    return merged


# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    final_response: str
    agent_usage: Annotated[dict, merge_usage]


# ── Specialist agents (ReAct: LLM + tool-calling loop) ──────────────────────
flight_react_agent = create_agent(
    llm,
    tools=[search_flights],
    system_prompt="You are a flight search specialist. Use the search_flights tool to find "
                  "flights matching the traveler's request, then summarize the best options "
                  "in plain text.",
)

hotel_react_agent = create_agent(
    llm,
    tools=[tavily_search],
    system_prompt="You are a hotel research specialist. Use the tavily_search tool to find "
                  "hotel options matching the traveler's request, then summarize the best "
                  "options in plain text.",
)


def flight_agent(state: TravelState) -> Command[Literal["supervisor"]]:
    tracker = AgentUsageTracker()
    result = flight_react_agent.invoke(
        {"messages": [HumanMessage(content=state["user_query"])]},
        config={"callbacks": [tracker]},
    )
    flight_text = result["messages"][-1].content
    return Command(
        goto="supervisor",
        update={
            "flight_results": flight_text,
            "messages": [AIMessage(content="Flight results fetched")],
            "agent_usage": {"flight_agent": tracker.as_dict()},
        },
    )


def hotel_agent(state: TravelState) -> Command[Literal["supervisor"]]:
    tracker = AgentUsageTracker()
    result = hotel_react_agent.invoke(
        {"messages": [HumanMessage(content=f"Best hotels for {state['user_query']}")]},
        config={"callbacks": [tracker]},
    )
    hotel_text = result["messages"][-1].content
    return Command(
        goto="supervisor",
        update={
            "hotel_results": hotel_text,
            "messages": [AIMessage(content="Hotel information fetched")],
            "agent_usage": {"hotel_agent": tracker.as_dict()},
        },
    )


def itinerary_agent(state: TravelState) -> Command[Literal["supervisor"]]:
    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    tracker = AgentUsageTracker()
    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner"),
            HumanMessage(content=prompt),
        ],
        config={"callbacks": [tracker]},
    )

    return Command(
        goto="supervisor",
        update={
            "itinerary": response.content,
            "messages": [response],
            "agent_usage": {"itinerary_agent": tracker.as_dict()},
        },
    )


def final_agent(state: TravelState) -> Command[Literal["supervisor"]]:
    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    tracker = AgentUsageTracker()
    response = llm.invoke(
        [HumanMessage(content=final_prompt)],
        config={"callbacks": [tracker]},
    )

    return Command(
        goto="supervisor",
        update={
            "final_response": response.content,
            "messages": [response],
            "agent_usage": {"final_agent": tracker.as_dict()},
        },
    )


# ── Supervisor ────────────────────────────────────────────────────────────
# To add a new specialist later (e.g. a car_rental_agent):
#   1. Build its tool + create_agent, like flight_agent/hotel_agent above.
#   2. Add it to the Router.next Literal and SUPERVISOR_PROMPT below.
#   3. Add it with graph.add_node(...) — no other wiring needed, the
#      supervisor routes to it dynamically.
class Router(BaseModel):
    next: Literal["flight_agent", "hotel_agent", "itinerary_agent", "final_agent", "FINISH"] = Field(
        description="Which agent should act next, or FINISH once final_agent has already produced a response."
    )


SUPERVISOR_PROMPT = """You are the supervisor of a multi-agent travel planning system.

Look at the user's request and the work completed so far, then decide which specialist
agent should act next.

Available agents:
- flight_agent: searches for flights. Call this if the request involves air travel/flights.
- hotel_agent: searches for hotels. Call this ONLY if the user explicitly asked about
  hotels or accommodation.
- itinerary_agent: builds a day-by-day itinerary from whatever research is available.
  Call this once every agent the request needs has finished.
- final_agent: writes the final response shown to the user. Call this after the
  itinerary is built.
- FINISH: call this once final_agent has already produced a response.

Rules:
- Never call an agent that has already completed its work.
- Never call hotel_agent if the user did not ask about hotels/accommodation.
- Don't call itinerary_agent until every agent the request needs has finished.
"""


def supervisor(state: TravelState) -> Command[Literal["flight_agent", "hotel_agent", "itinerary_agent", "final_agent", "__end__"]]:
    status = f"""User request: {state['user_query']}

Progress:
- flight_agent: {"done" if state.get("flight_results") else "not done"}
- hotel_agent: {"done" if state.get("hotel_results") else "not done"}
- itinerary_agent: {"done" if state.get("itinerary") else "not done"}
- final_agent: {"done" if state.get("final_response") else "not done"}
"""

    tracker = AgentUsageTracker()
    decision = llm.with_structured_output(Router).invoke(
        [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=status),
        ],
        config={"callbacks": [tracker]},
    )

    goto = END if decision.next == "FINISH" else decision.next
    return Command(goto=goto, update={"agent_usage": {"supervisor": tracker.as_dict()}})


graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")

_conn = psycopg.connect(DATABASE_URL, autocommit=True)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)


def check_postgres_connection(conn: Optional[psycopg.Connection] = None, dsn: Optional[str] = None, timeout: float = 5.0) -> bool:
    if conn is None and dsn is None:
        raise ValueError("Provide either `conn` or `dsn`")
    created = False
    try:
        if conn is None:
            conn = psycopg.connect(dsn, connect_timeout=timeout)
            created = True
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        cur.close()
        return True
    except Exception:
        return False
    finally:
        if created and conn:
            conn.close()


if __name__ == "__main__":
    dsn = os.getenv("DATABASE_URL")
    if not check_postgres_connection(dsn=dsn):
        raise SystemExit("Failed to connect to the database.")

    print("Successfully connected to the database.")

    config = {
        "configurable": {
            "thread_id": "user_gull"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "final_response": "",
            "agent_usage": {}
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")
    print(result["final_response"])

    print("\nLLM USAGE BY AGENT:\n")
    print(f"{'Agent':<16}{'Calls':>8}{'Input':>10}{'Output':>10}{'Total':>10}")
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for agent, stats in result["agent_usage"].items():
        print(f"{agent:<16}{stats['calls']:>8}{stats['input_tokens']:>10}{stats['output_tokens']:>10}{stats['total_tokens']:>10}")
        for key in totals:
            totals[key] += stats[key]
    print("-" * 54)
    print(f"{'TOTAL':<16}{totals['calls']:>8}{totals['input_tokens']:>10}{totals['output_tokens']:>10}{totals['total_tokens']:>10}")
