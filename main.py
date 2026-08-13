import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated, Optional
import psycopg

from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search

import psycopg
from langgraph.graph import StateGraph, START , END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (AnyMessage,HumanMessage, SystemMessage, AIMessage)
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    
)



# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

# Flight Agent
def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content=f"Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Itinerary Agent
def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Final Response Agent
def final_agent(state: TravelState):

    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

_conn = psycopg.connect(DATABASE_URL)
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
        print("Failed to connect to the database.")
    else:
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
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)

    # # Initialize the PostgresSaver with the connection string
    # saver = PostgresSaver(dsn=dsn)

    # # Create a StateGraph instance
    # graph = StateGraph(llm=llm, saver=saver)

    # # Example usage of the graph
    # graph.add_state(START, SystemMessage(content="You are a helpful assistant."))
    # graph.add_state("search_flights", HumanMessage(content="Search for flights from New York to London."))
    # graph.add_state("tavily_search", HumanMessage(content="Search for information about AI tools."))

    # # Define transitions
    # graph.add_transition(START, "search_flights", condition=lambda msg: "flight" in msg.content.lower())
    # graph.add_transition(START, "tavily_search", condition=lambda msg: "AI" in msg.content.lower())

    # # Execute the graph
    # graph.execute()