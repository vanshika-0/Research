# Research Desk

Research Desk is a multi-agent research assistant. A Next.js interface sends a question to a FastAPI backend, which uses LangGraph to coordinate specialist research agents, collect web, academic-paper, news, and YouTube results, and generate a sourced Markdown report with an LLM.

## Project structure

```text
backend/
  app.py           FastAPI application and /api/research endpoint
  main.py          LangGraph workflow, agents, and report generation
  MCP_servers.py   Local MCP tools for web, papers, news, and YouTube search
tripmate-next/
  app/             Next.js App Router frontend
```

## End-to-end project flow

```text
User enters a question in the browser
          │
          ▼
Next.js /dashboar page
  - sends POST /api/research
  - saves the latest query/report in sessionStorage
          │
          ▼
FastAPI backend (backend/app.py)
  - validates {"message": "..."}
  - rejects empty input with 400
  - calls run_research_agent()
          │
          ▼
LangGraph workflow (backend/main.py)
  1. Guardrail agent checks whether the request is suitable
  2. Supervisor chooses the required specialist agents
  3. Selected agents run in parallel using LangGraph Send
          │
          ├── Web research agent
          │     └── Tavily through the local MCP server
          ├── Paper research agent
          │     └── OpenAlex, with article/PDF extraction when available
          ├── News research agent
          │     └── NewsAPI
          └── YouTube research agent
                └── YouTube Data API and transcript lookup
          │
          ▼
Research fan-in
  - combines web, paper, news, and video results
          │
          ▼
Final analysis agent
  - serializes all collected research
  - generates the report directly for small context
  - chunks and summarizes large context before final synthesis
  - preserves source names, titles, and URLs
          │
          ▼
FastAPI serializes the result
  - returns answer, source data, selected agents, and metadata
          │
          ▼
Next.js renders the Markdown report
  - source links open from the report
  - returning to the tab restores the saved report
```

### Request lifecycle details

1. The user submits a question from `tripmate-next/app/dashboar/page.js`. The frontend sends JSON to `POST /api/research` and displays the returned `answer` as Markdown.
2. `backend/app.py` validates the request with the `ResearchRequest` Pydantic model. It calls `run_research_agent(user_input=...)` and converts unexpected exceptions into an HTTP `500` response. Full tracebacks are logged by the backend.
3. `backend/main.py` creates the compiled LangGraph once and reuses it for later requests. The graph starts at `guardrail_agent`.
4. The guardrail either ends the graph with a blocked response or passes the request to `supervisor_agent`. If the supervisor cannot route the request, the graph returns a controlled supervisor-error response.
5. The supervisor selects from `web_research_agent`, `paper_research_agent`, `news_research_agent`, and `youtube_research_agent`. If it selects none, all four research agents run as a fallback.
6. Each specialist asks the local `backend/MCP_servers.py` process for data. The MCP server is launched over stdio by `MultiServerMCPClient`, so the frontend never calls search providers directly.
7. All selected agents fan in to `research_complete`, then `final_agent` builds the final report. Reports use the collected research only and include summary, findings, analysis, academic research, recent news, and sources.
8. The backend returns the serialized result. The frontend stores the query and answer in tab-scoped `sessionStorage`, allowing the report to survive source-link navigation, browser back navigation, and refreshes in the same tab.

### Failure and fallback paths

- Blocked requests end at `guardrail_blocked`.
- Supervisor failures end at `supervisor_error` with a user-safe message.
- Individual search-agent failures return empty or unavailable source data so the rest of the workflow can continue.
- Large research contexts are split into chunks and summarized before final report generation.
- Backend exceptions are logged with a traceback; clients receive a generic error unless `DEBUG=true` is set locally.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- API keys for Groq, Tavily, and NewsAPI
- A YouTube API key if YouTube search is enabled
- MongoDB is optional for the current research workflow, but the backend can be configured with a MongoDB URI

## Configuration

Create a `.env` file in the project root. Do not commit this file or publish API keys.

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
NEWS_API_KEY=your_newsapi_key
YOUTUBE_API_KEY=your_youtube_api_key

# Optional
MONGO_URI=mongodb://localhost:27017
FRONTEND_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

`GROQ_API_KEY`, `TAVILY_API_KEY`, and `NEWS_API_KEY` are required when the backend starts. `NEXT_PUBLIC_API_URL` is read by the frontend and should contain the backend origin; the frontend appends `/api/research` automatically.

## Install dependencies

### Backend

Using the existing `uv` project:

```bash
uv sync
```

Or install from the requirements file:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Frontend

```bash
cd tripmate-next
npm install
```

## Run locally

Start the backend from the repository root:

```bash
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```bash
cd tripmate-next
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The backend health check is available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## API

### `POST /api/research`

Request:

```json
{
  "message": "How does retrieval-augmented generation work?"
}
```

Successful responses include the generated answer, selected agents, research constraints, source results, and metadata:

```json
{
  "success": true,
  "answer": "## Findings\\n...",
  "selected_agents": ["web_research_agent", "paper_research_agent"],
  "usage": null
}
```

An empty message returns `400`. Invalid request bodies return `422`. Unexpected research failures return `500`; the backend logs the full traceback while keeping the default client error generic. Set `DEBUG=true` locally if you need the exception text in the response.

## Frontend behavior

The report is rendered as Markdown, including source links and tables. The latest query and report are temporarily saved in `sessionStorage`, so they survive opening a source link, browser back navigation, or a refresh in the same tab. Starting a new research request replaces the saved report.

## Useful commands

```bash
# Backend tests/scripts
python test.py

# Frontend development
cd tripmate-next
npm run dev

# Frontend production build
npm run build
npm run start
```

## Troubleshooting

- If the backend fails during import, check that the required API keys are present in `.env`.
- If the browser cannot reach the API, verify that the backend is running on port `8000` and that `NEXT_PUBLIC_API_URL` points to the backend origin.
- If the browser reports a CORS error, add the frontend origin to `FRONTEND_ORIGINS` and restart the backend.
- If research returns a temporary fallback response, check the backend logs for Groq, Tavily, NewsAPI, or MCP connection errors.
