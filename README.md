# ✈️ AI Travel Booking System

A multi-agent AI travel planner built with **LangGraph**, using a **supervisor pattern**
to dynamically decide which specialist agents to run based on what the user actually
asks for (e.g. flights only, or flights + hotel).

## How it works

A supervisor node looks at the user's request and the work done so far, then routes
to the right specialist agent — instead of always running every agent in a fixed order.

```
START → supervisor ⇄ flight_agent      (only if flights are requested)
                    ⇄ hotel_agent       (only if hotels are requested)
                    ⇄ itinerary_agent   (once research is done)
                    ⇄ final_agent       (writes the response shown to the user)
                    → FINISH → END
```

- **supervisor** — an LLM call with structured output (`Router`) that decides the next
  agent to call, or `FINISH`. Never calls `hotel_agent` unless hotels were requested.
- **flight_agent** / **hotel_agent** — real ReAct agents (`langchain.agents.create_agent`):
  the LLM reasons about the request and decides how to call its tool
  (`search_flights` / `tavily_search`), instead of a hardcoded function call.
- **itinerary_agent** — synthesizes flight + hotel results into a day-by-day plan.
- **final_agent** — writes the final response shown to the user.

Conversation state (per user) is persisted in **PostgreSQL** via LangGraph's
`PostgresSaver` checkpointer, keyed by a `thread_id`.

Adding a new specialist later (e.g. a `car_rental_agent`) just means: build its tool +
`create_agent`, add it to the `Router` and supervisor prompt, and `add_node(...)` —
no other wiring needed. See the comment in `main.py` above the `Router` class.

## Project structure

```
main.py                    LangGraph pipeline: state, agents, supervisor, build_app()
frontend.py                 Streamlit UI
tools/flight_tool.py         AviationStack flight search tool
tools/tavily_tool.py         Tavily web search tool (used for hotels)
travel_plans/                Auto-saved trip plans (created at runtime)
pyproject.toml / uv.lock     Dependencies (managed by uv)
```

## Setup

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Running it

### Streamlit UI (recommended)

```bash
uv run streamlit run frontend.py
```

On first load, the sidebar's **API Configuration** section is empty. Enter:

- **Tavily API Key** — used by `hotel_agent` for hotel search
- **Groq API Key** — powers all the LLM calls (`llama-3.3-70b-versatile`)
- **AviationStack API Key** — used by `flight_agent` for flight search
- **Database URL** — a PostgreSQL connection string, used for conversation checkpointing

Click **Connect**. This validates the database connection and builds the agent graph
for that session — nothing is written to `.env`, the keys only live in memory for the
running session. The trip-planning form appears once connected.

### CLI

```bash
uv run python main.py
```

Reads `GROQ_API_KEY` and `DATABASE_URL` from `.env` directly (no sidebar step),
prompts for a travel request, and prints the final plan plus a per-agent
token-usage breakdown.

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `GROQ_API_KEY` | `main.py` | LLM calls (supervisor, agents, itinerary/final synthesis) |
| `TAVILY_API_KEY` | `tools/tavily_tool.py` | Hotel/web search |
| `AVIATIONSTACK_API_KEY` | `tools/flight_tool.py` | Flight search |
| `DATABASE_URL` | `main.py` | PostgreSQL connection string for conversation checkpointing |

For the CLI, put these in `.env`. For the Streamlit UI, enter them in the sidebar instead.

## Observability

Every LLM call (including the supervisor's own routing calls and the internal
tool-decide/summarize calls inside each ReAct agent) is tracked per-agent via a
callback-based usage tracker (`AgentUsageTracker` in `main.py`), showing real call
counts and input/output/total token usage — visible in the Streamlit UI's
"Token Usage by Agent" table and in the CLI's printed summary.
