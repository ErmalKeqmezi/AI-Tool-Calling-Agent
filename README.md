# AI Tool-Calling Agent (from scratch)

A small, readable AI assistant that **decides for itself** when to call tools, runs them in
Python, feeds the results back to the LLM, and repeats until it can answer. Built on the
Anthropic Messages API with its native tool calling. No agent frameworks, no RAG, no
embeddings, no vector databases. The loop is written by hand so you can see every step.

## How to use it

### 1. Set up (once)

You need Python 3.10+ and an Anthropic API key (get one at https://console.anthropic.com).

```bash
git clone https://github.com/ErmalKeqmezi/AI-Tool-Calling-Agent.git
cd AI-Tool-Calling-Agent
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
copy .env.example .env            # macOS / Linux: cp .env.example .env
```

Open `.env` and set your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Start the app

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501. If the sidebar shows **● Online**, the agent is ready.

### 3. Chat with the agent

- Type a question in the box at the bottom and press **Enter**, or click one of the example prompts.
- The agent decides on its own whether it needs a tool. While it works, you'll see *🤔 Agent is thinking…* or *🔧 Using Calculator…*.
- Every tool the agent used appears above its answer as a collapsible block (`✓ calculate · expression: 25 * 17 · Completed`). Click it to see the exact arguments and the result.
- Ask follow-up questions naturally. The agent remembers the conversation ("What's the weather in Berlin?" → "And tomorrow?").
- Actions that change something, like creating or deleting a file, are **never run automatically**. The agent shows what it wants to do and waits for you to click **✅ Approve** or **❌ Deny**. Files are only written inside the `workspace/` folder.
- **Clear conversation** in the sidebar starts a fresh chat and wipes the agent's memory.

Things to try:

| Ask | Tools it will use |
|---|---|
| What is 25 × 17? | `calculate` |
| What time is it in Tokyo? | `get_current_time` |
| What's the weather in Berlin? Then: "What about tomorrow?" | `get_weather` (twice, using memory) |
| What is 25 × 17 and what time is it in Tokyo? | `calculate` + `get_current_time` |
| Search the web for the latest Python release. | `search_web` |
| Create a file called notes.txt with a haiku about Python. | `create_file` (asks for approval) |

### Other ways to run it

| Interface | Command |
|---|---|
| Terminal chat | `python -m app.main` (add `-v` to see every tool call in the logs, `/reset` clears memory, `/quit` exits) |
| HTTP API | `uvicorn app.api.server:create_app --factory --reload`, then `POST /chat` (see [API](#api)) |
| Tests (no API key needed) | `pytest` |

Optional: set `TAVILY_API_KEY` in `.env` for full web search results (a free key is available at https://tavily.com). Without it, search falls back to DuckDuckGo instant answers.

## 1. What it does

```
you> What is 25 * 17 and what time is it in Tokyo?
assistant> 25 × 17 = 425. In Tokyo it's currently 18:42 (Monday, UTC+09:00).
```

Behind that answer the LLM asked for two tools (`calculate`, `get_current_time`). Python ran
them, and the LLM wrote the final answer from their results.

Built-in tools:

| Tool | Permission | What it does |
|---|---|---|
| `calculate(expression)` | READ | Safe arithmetic (AST allow-list, no `eval`) |
| `get_current_time(timezone)` | READ | Current time in an IANA timezone |
| `get_weather(city, days_ahead)` | READ | Current weather / forecast via Open-Meteo (no key) |
| `search_web(query, max_results)` | READ | Tavily if `TAVILY_API_KEY` is set, otherwise DuckDuckGo Instant Answers |
| `create_file(filename, content, overwrite)` | **WRITE** | Writes a file inside `workspace/` |
| `delete_file(filename)` | **WRITE** | Deletes a file inside `workspace/` |

## 2. What is tool calling?

An LLM can only produce text. With tool calling, you send it a list of **tool definitions**
(name, description, JSON Schema for the arguments). The model can then reply with a
structured request instead of prose:

```json
{"type": "tool_use", "id": "toolu_01", "name": "calculate", "input": {"expression": "25 * 17"}}
```

The model **does not run anything**. Your code decides whether to run it, runs it, and
sends the result back:

```json
{"type": "tool_result", "tool_use_id": "toolu_01", "content": "{\"success\": true, \"result\": {\"value\": 425}}"}
```

The model reads the result and either asks for more tools or writes the final answer.

## 3. Architecture

```
            ┌──────────────┐        ┌────────────────┐
  HTTP ───► │  API layer   │        │  CLI (main.py) │ ◄─── terminal
            │ api/server.py│        └───────┬────────┘
            └──────┬───────┘                │
                   └──────────┬─────────────┘
                              ▼
                ┌───────────────────────────┐
                │  Agent  (agent/agent.py)  │  conversation memory
                │  • the loop               │  permission gate (READ / WRITE)
                └──────┬──────────────┬─────┘
                       │              │
         messages +    │              │ tool name + arguments
         tool schemas  ▼              ▼
        ┌────────────────────┐   ┌─────────────────────────────┐
        │ LLMClient          │   │ ToolExecutor                │
        │ (llm/client.py)    │   │ (agent/executor.py)         │
        │ Anthropic SDK,     │   │ lookup → validate → run →   │
        │ error mapping      │   │ catch → structured result   │
        └─────────┬──────────┘   └──────────────┬──────────────┘
                  ▼                             ▼
          Anthropic Messages API     ┌──────────────────────────┐
                                     │ ToolRegistry (tools/)    │
                                     │ name → Tool(schema, fn,  │
                                     │            permission)   │
                                     └──────────────────────────┘
```

```
ai-tool-agent/
├── app/
│   ├── main.py                 # CLI (python -m app.main)
│   ├── ui/presenters.py        # formatting + friendly errors for the Streamlit UI
│   ├── logging_config.py       # log format + secret redaction
│   ├── config/settings.py      # env-based settings
│   ├── llm/client.py           # LLMClient, LLMResponse, ToolCall, LLMError hierarchy
│   ├── agent/
│   │   ├── agent.py            # the agent loop, permissions, memory
│   │   └── executor.py         # ToolExecutor
│   ├── tools/
│   │   ├── __init__.py         # build_registry(): the one place tools are registered
│   │   ├── base.py             # Tool, Permission, ToolRegistry, ToolError, ok()/fail()
│   │   ├── validation.py       # tiny JSON-Schema validator
│   │   ├── calculator.py  time.py  weather.py  search.py  files.py
│   └── api/server.py           # FastAPI: POST /chat
├── app.py                      # Streamlit UI (streamlit run app.py)
├── .streamlit/config.toml      # dark theme
├── tests/                      # pytest with a scripted FakeLLM (+ optional live tests)
├── .env.example  requirements.txt  README.md
```

Each layer only knows about the one below it. The API knows nothing about tools, and the
executor knows nothing about the LLM. In tests, the LLM is swapped for a script.

## 4. The agent loop

`Agent._run_loop()` in [app/agent/agent.py](app/agent/agent.py):

```python
for iteration in range(max_iterations):
    response = llm.chat(messages, tools=registry.schemas(), system=system_prompt)

    if not response.tool_calls:                       # no tool wanted → done
        messages.append({"role": "assistant", "content": response.content})
        return final_answer

    messages.append({"role": "assistant", "content": response.content})   # the tool request
    results = {call.id: executor.execute(call.name, call.arguments) for call in response.tool_calls}
    messages.append({"role": "user", "content": [tool_result blocks...]}) # the results
    # loop → the LLM sees the results and decides what's next
```

The real code adds:
- **Permission gate.** WRITE tools are held back and the user is asked first (§9).
- **Iteration cap.** `AGENT_MAX_ITERATIONS` stops a runaway loop.
- **Stop reasons.** `max_tokens` (a truncated tool call is never run), `refusal`, and `pause_turn` are handled.
- **Valid history.** After any failure the history still alternates user/assistant, so the next turn works.

The loop never contains `if "weather" in message`. **The LLM picks the tools.** The loop
only runs what was requested.

## 5. How tools are defined

A tool is a `Tool` dataclass ([app/tools/base.py](app/tools/base.py)):

```python
CALCULATOR_TOOL = Tool(
    name="calculate",
    description="Evaluate an arithmetic expression exactly. Use this for any calculation ...",
    parameters={                                  # JSON Schema
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "...", "minLength": 1}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    function=calculate,                           # the Python that actually runs
    permission=Permission.READ,
)
```

`Tool.to_llm_schema()` converts it to Anthropic's wire format (`name`, `description`,
`input_schema`). The **description matters most**: it is how the LLM decides *when* to use
the tool.

Tools return plain data or raise `ToolError(message, error_type)`. The executor wraps
the outcome in one of two shapes:

```json
{"success": true,  "result": {"expression": "25 * 17", "value": 425}}
{"success": false, "error": {"type": "invalid_timezone", "message": "Unknown timezone 'Tokyo'. Did you mean ['Asia/Tokyo']?"}}
```

## 6. How tools are registered

[app/tools/__init__.py](app/tools/__init__.py) is the only place tools are listed:

```python
registry.register(CALCULATOR_TOOL)
registry.register(TIME_TOOL)
registry.register(make_weather_tool(settings.tool_http_timeout))
...
```

The registry maps `name → Tool`. The executor looks names up there, so there's no if/else
chain. `registry.schemas()` gives the list sent to the LLM on every call.

## 7. How the executor works

`ToolExecutor.execute(name, arguments, approved=False)`
([app/agent/executor.py](app/agent/executor.py)) never trusts the LLM:

1. **Tool exists?** Unknown or non-string names return `unknown_tool` with the list of valid names.
2. **Permission.** A WRITE tool without `approved=True` returns `permission_required`. Only the agent sets that flag, after the user says yes.
3. **Arguments parse?** A JSON string is decoded. Anything that isn't an object returns `malformed_arguments`.
4. **Arguments valid?** They are checked against the schema: required fields, types, ranges, lengths, no extras. Failures return `invalid_arguments` with a precise message.
5. **Run.** The call is timed. `ToolError` becomes a structured error, and any other exception becomes `execution_error`, so a buggy tool can't crash the agent.
6. **Truncate** huge results, then log and return.

## 8. Multi-step tool calls

Two patterns both fall out of the loop naturally:

- **Parallel.** One LLM response contains several `tool_use` blocks (`calculate` + `get_current_time`). All are executed, and all results go back in **one** user message.
- **Sequential / dependent.** The LLM calls `calculate("25*17")`, sees `425`, then calls `calculate("425/5")` on the next iteration. Each result is in the history, so later calls can build on earlier ones.

```
User ─► LLM ─► calculate("25*17") ─► 425 ─► LLM ─► calculate("425/5") ─► 85 ─► LLM ─► "…is 85"
```

## 9. Permissions

READ tools run automatically. WRITE tools pause the loop:

```
you> create a file hello.txt that says hi
assistant> This action requires your confirmation:
  - create_file({"filename": "hello.txt", "content": "hi"})
Reply 'yes' to proceed or 'no' to cancel.
you> yes
assistant> Created hello.txt.
```

- The permission comes from the **registry**, not from anything the LLM says.
- The confirmation answer is parsed by **Python**, not by the LLM. "yes", "ok" and similar approve. Anything else declines.
- If the user replies with something else ("actually, what's the weather?"), the action is declined *and* the new message is forwarded to the LLM.
- A declined call is reported to the LLM as a `permission_denied` tool result, so it can respond appropriately.
- The executor refuses WRITE tools without `approved=True` as a second line of defense.
- File tools are confined to `workspace/`. Paths like `../x` are rejected.

## 10. Error handling

| Failure | Where handled | What happens |
|---|---|---|
| Missing API key | `Settings.require_api_key` | Clear message at startup |
| Timeout / network / 5xx / overloaded | `LLMClient.chat` | SDK retries (`LLM_MAX_RETRIES`), then `LLMTimeoutError` / `LLMUnavailableError` |
| Rate limit (429) | `LLMClient.chat` | SDK backs off and retries, then `LLMRateLimitError` |
| Bad key / model / request | `LLMClient.chat` | `LLMAuthError` / `LLMBadRequestError` |
| Unusable response | `LLMClient._normalize` | `LLMInvalidResponseError` |
| Any `LLMError` in the loop | `Agent._run_loop` | Friendly reply with `error` set; history stays valid |
| Unknown tool, bad or missing args, malformed JSON | `ToolExecutor` | Structured error sent to the LLM (it can self-correct) |
| Tool raises | `ToolExecutor` | `execution_error` sent to the LLM |
| Tool's upstream API times out / 429 | the tool | `ToolError("timeout" / "rate_limited")` |
| Endless tool loop | `Agent` | Stops after `AGENT_MAX_ITERATIONS` |

## 11. Creating a new tool

1. Create `app/tools/my_tool.py`:

```python
from app.tools.base import Permission, Tool, ToolError

def convert_currency(amount: float, frm: str, to: str) -> dict:
    if frm == to:
        raise ToolError("Currencies must differ.", "invalid_input")
    ...
    return {"amount": amount, "from": frm, "to": to, "converted": converted}

CURRENCY_TOOL = Tool(
    name="convert_currency",
    description="Convert an amount between currencies using today's rate.",
    parameters={
        "type": "object",
        "properties": {
            "amount": {"type": "number", "minimum": 0},
            "frm": {"type": "string", "description": "ISO code, e.g. 'EUR'"},
            "to": {"type": "string", "description": "ISO code, e.g. 'USD'"},
        },
        "required": ["amount", "frm", "to"],
        "additionalProperties": False,
    },
    function=convert_currency,
    permission=Permission.READ,   # WRITE if it has side effects
)
```

2. Register it in `build_registry()`: `registry.register(CURRENCY_TOOL)`.

That's all. The agent, executor and API don't change.

## 12. Memory

Short-term only. `Agent.messages` holds the conversation in API format, including tool
calls and results, and is sent in full on every call. That is why "What about tomorrow?"
after a Berlin weather question works: the model sees "Berlin" earlier in the history and
calls `get_weather(city="Berlin", days_ahead=1)`.

The history is capped at `AGENT_MAX_HISTORY_TURNS` user turns. Trimming only cuts at the
start of a user turn, never between a `tool_use` and its `tool_result`. No RAG and no
vector store.

## 13. Logging

```
11:33:51 INFO    agent.loop | USER: What is 25 * 17?
11:33:51 INFO    agent.loop | LLM RESPONSE #1: stop_reason=tool_use tools=['calculate'] text=''
11:33:51 INFO    agent.executor | TOOL SELECTED: calculate
11:33:51 INFO    agent.executor | ARGUMENTS: {"expression": "25 * 17"}
11:33:51 INFO    agent.executor | TOOL RESULT (calculate, 0 ms): {"success": true, "result": {"expression": "25 * 17", "value": 425}}
11:33:51 INFO    agent.loop | LLM RESPONSE #2: stop_reason=end_turn tools=[] text='25 × 17 = 425'
11:33:51 INFO    agent.loop | FINAL: 25 × 17 = 425
```

Keys are never logged. Settings hide them from `repr`, and a logging filter redacts
anything that looks like `sk-ant-…`, `tvly-…` or `api_key=…`. SDK/HTTP loggers are kept at
WARNING.

## 14. Running it

```bash
cd ai-tool-agent
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env            # then put your ANTHROPIC_API_KEY in .env
```

| What | Command |
|---|---|
| **Web UI (Streamlit)** | `streamlit run app.py` → http://localhost:8501 |
| Plain LLM chat, no tools (Phase 1) | `python -m app.main --no-tools` |
| Full agent in the terminal | `python -m app.main` (add `-v` for DEBUG logs) |
| HTTP API | `uvicorn app.api.server:create_app --factory --reload` |
| Offline tests (no key needed) | `pytest` |
| Live tests against the real model | `pytest tests/test_live.py -v` (needs the key) |

### Streamlit UI

`app.py` (project root) is a pure UI layer on top of the same `Agent`:

```
Streamlit (app.py) ─► Agent.chat(message, on_event=...) ─► LLM ─► tool call ─► ToolExecutor
        ▲                     │                                  ─► Python tool ─► result ─► LLM
        └── AgentEvent ◄──────┘  thinking / tool_start / tool_end / confirmation_required
        └── AgentReply ◄──────── final text + tool_calls (name, arguments, result)
```

- **Session state.** `st.session_state.agent` holds one `Agent` per browser tab, including its LLM-format memory. `st.session_state.messages` is the UI's copy of the chat: `{"role": "user", "content", "display"}` and `{"role": "assistant", "content", "tool_calls", "error", "pending_confirmation"}`. On every rerun the page is redrawn from that list.
- **Live progress.** `Agent.chat()` takes an optional `on_event` callback. The UI uses it to update an `st.status` ("🤔 Agent is thinking…" → "🔧 Using Calculator…") and to add one `st.expander` per tool call as each one finishes. Only events the agent actually emits are shown.
- **Tool calls.** Each expander shows the tool name, a one-line argument summary and ✓ Completed / ✗ Failed / ⊘ Declined. Expanding it shows the arguments and the result as JSON.
- **Confirmations.** WRITE tools show ✅ Approve / ❌ Deny buttons, which send "yes" / "no" to the agent exactly as if typed.
- **Errors.** LLM failures appear as friendly warnings ("⚠️ Unable to contact the AI model…"). Tool failures show the tool's structured error message, never a stack trace. Details stay in the server log.
- **Theme.** Dark, set in `.streamlit/config.toml`. The sidebar tool list is read from the registry.

### API

`POST /chat`

```json
{"message": "What is 25 * 17?", "session_id": null}
```

```json
{
  "response": "25 × 17 = 425",
  "session_id": "3f2a…",
  "tool_calls": [{"name": "calculate", "arguments": {"expression": "25 * 17"}, "result": {"success": true, "result": {"expression": "25 * 17", "value": 425}}}],
  "pending_confirmation": null,
  "error": null
}
```

Send the returned `session_id` with the next message to continue the conversation (memory
and confirmations are per session). `DELETE /sessions/{id}` forgets a session, and
`GET /health` is a liveness check. Sessions live in memory, so they reset when the server
restarts.

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"What time is it in Tokyo?\"}"
```

## 15. Example conversations

**No tool**
```
you> Hello
assistant> Hi! How can I help you today?            (0 tool calls)
```

**Tool error → self-correction**
```
you> What time is it in Tokyo?
  LLM → get_current_time("Tokyo")      → error: Unknown timezone… Did you mean ['Asia/Tokyo']?
  LLM → get_current_time("Asia/Tokyo") → 18:42
assistant> It's 18:42 in Tokyo.
```

**Follow-up using memory**
```
you> What's the weather in Berlin?
assistant> 18°C and mainly clear.
you> What about tomorrow?
  LLM → get_weather(city="Berlin", days_ahead=1)
assistant> Tomorrow: 15°C high, 8°C low, 80% chance of rain.
```

## Configuration

All settings come from environment variables (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `LLM_MODEL` | `claude-opus-5` | Any Claude model with tool use |
| `LLM_MAX_TOKENS` | `16000` | Per-response output cap |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `60` / `2` | HTTP timeout and SDK retries |
| `LLM_REFUSAL_FALLBACK` | `true` | If the model declines a request on safety grounds, the API retries it on a fallback model (`fallbacks: "default"`) |
| `AGENT_MAX_ITERATIONS` | `10` | LLM round-trips per user message |
| `AGENT_MAX_HISTORY_TURNS` | `20` | Short-term memory size |
| `TAVILY_API_KEY` | — | Optional, for full web search |
| `FILE_SANDBOX_DIR` | `./workspace` | Where the file tools may write |

## Author

**Ermal Keqmezi**  
Software Developer & AI Engineer

## License

This project is available for educational and portfolio purposes.
