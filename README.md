

# Getting Started with ROMA: Run Your First Multi-Agent Task in Under 5 Minutes

ROMA (Recursive Open Meta-Agents) is a multi-agent framework that breaks complex tasks into subtasks and solves them using a pipeline of specialized agents: **Atomizer → Executor → Aggregator**. This tutorial walks you through installation and your first working task from scratch.

---

## Prerequisites

- Python 3.12+
- An API key from Anthropic, OpenAI, or any provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers)
- `uv` (recommended) or `pip`

---

## Step 1: Install ROMA

Clone the repo and install in editable mode:

```bash
git clone https://github.com/sentient-agi/ROMA.git
cd ROMA
python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -e .
```

> **Why editable mode?** `roma-dspy` is not yet published to PyPI, so you install directly from the local source. The `-e` flag means changes to the source are reflected immediately without reinstalling.

---

## Step 2: Set Your API Key

ROMA uses [LiteLLM](https://docs.litellm.ai) under the hood, which supports all major providers. Set the key for whichever provider you're using:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# OpenRouter (routes to any model)
export OPENROUTER_API_KEY="sk-or-..."
```

---

## Step 3: Run Your First Task

The real power of ROMA shows up with multi-step tasks that require research, analysis, and synthesis — the kind of work that benefits from being broken into parallel subtasks handled by specialized agents.

Create a file `hello_roma.py`:

```python
from roma_dspy.core.engine.solve import solve
from roma_dspy.config.schemas.root import ROMAConfig
from roma_dspy.config.schemas.base import RuntimeConfig, LLMConfig
from roma_dspy.config.schemas.agents import AgentsConfig, AgentConfig

agent_cfg = AgentConfig(llm=LLMConfig(model="anthropic/claude-haiku-4-5-20251001", max_tokens=4000))

config = ROMAConfig(
    runtime=RuntimeConfig(timeout=700, max_depth=2),
    agents=AgentsConfig(
        atomizer=agent_cfg,
        planner=agent_cfg,
        executor=agent_cfg,
        aggregator=agent_cfg,
        verifier=agent_cfg,
    )
)

task = """
A developer is deciding between asyncio, threading, and multiprocessing for a Python project.
Help them make the right choice by:
1. Explaining what each approach is best suited for (I/O-bound vs CPU-bound workloads)
2. Describing the key trade-offs of each (complexity, overhead, GIL)
3. Applying this to two scenarios: (a) a web scraper hitting 100 URLs, (b) resizing 1000 images
4. Giving a final verdict for each scenario with a short code example
"""

result = solve(task, config=config)
print(result.result)
```

Run it:

```bash
python hello_roma.py
```

ROMA will **decompose this into subtasks**, run them through the agent pipeline, and synthesize a structured answer. You'll see it spawn multiple subtasks — in testing, this task produced 9 first-level subtasks (one per distinct question) that the Planner ordered and the Executor handled in parallel before the Aggregator combined the results.

> **Note on the default model:** ROMA defaults to `gpt-4o-mini`. If you're using Anthropic, you'll need to set `ANTHROPIC_API_KEY` and specify the model with the `anthropic/` prefix as shown above.

---

## Step 4: Understand the Key Config Parameters

The `ROMAConfig` has three parameters that most directly affect behavior:

```python
config = ROMAConfig(
    runtime=RuntimeConfig(
        timeout=700,    # total wall-clock timeout in seconds (must exceed per-agent timeout, default 600s)
        max_depth=2,    # how many levels of subtask decomposition to allow (1–5 recommended)
    ),
    agents=AgentsConfig(
        atomizer=AgentConfig(llm=LLMConfig(
            model="anthropic/claude-haiku-4-5-20251001",
            max_tokens=4000,   # per-call token budget; increase for tasks with long outputs
            temperature=0.7,
        )),
        # planner, executor, aggregator, verifier follow the same pattern
        ...
    )
)
```

**`max_depth`** controls how aggressively ROMA decomposes tasks:
- `max_depth=1` — no decomposition, single agent handles everything
- `max_depth=2` — one level of subtasks (recommended starting point)
- `max_depth=3+` — deeper trees for very complex research tasks; increases latency

**`max_tokens`** — set this higher (3000–4000) if subtasks are producing truncated results.

**`timeout`** — must be greater than the per-agent LLM timeout (default: 600s). A safe value is `700`.

### Model naming

ROMA uses LiteLLM for provider routing. Model names must include the **provider prefix**:

| Provider | Example model ID |
|----------|-----------------|
| Anthropic | `anthropic/claude-haiku-4-5-20251001` |
| OpenAI | `openai/gpt-4o-mini` |
| Google | `gemini/gemini-2.0-flash` |
| OpenRouter | `openrouter/anthropic/claude-3.5-sonnet` |

If you omit the prefix (e.g. just `claude-haiku-4-5-20251001`), LiteLLM won't know which provider to route to and will raise an error.

### Timeout configuration

The `runtime.timeout` must exceed the per-agent timeout (default: 600s). A safe starting value is `RuntimeConfig(timeout=700)`.

---

## Step 5: Understand the Output

`solve()` returns a `TaskNode` — ROMA's internal representation of the task tree. The useful parts:

```python
result = solve("Your task", config=config)

print(result.result)      # the final answer (str)
print(result.status)      # COMPLETED, FAILED, etc.
print(result.goal)        # the original task
print(result.metrics)     # token usage, timing, etc.
```

---

## What Happens Under the Hood

When you call `solve()`, ROMA runs a state machine through these stages:

```
Your task
   │
   ▼
Atomizer      Breaks the task into atomic subtasks
   │
   ▼
Planner       Orders subtasks and resolves dependencies
   │
   ▼
Executor      Runs each subtask (with optional tools)
   │
   ▼
Aggregator    Synthesizes subtask results into a final answer
   │
   ▼
Verifier      Checks the answer against the original goal
```

For simple tasks, ROMA may skip decomposition and go straight to execution. For complex tasks it creates a full DAG of subtasks.

---

## Adding Tools

The Executor can use tools to complete tasks. Enable the built-in toolkits via config:

```python
from roma_dspy.config.schemas.agents import AgentConfig, ToolkitConfig

executor_cfg = AgentConfig(
    llm=LLMConfig(model="anthropic/claude-haiku-4-5-20251001"),
    toolkits=[
        ToolkitConfig(class_name="CalculatorToolkit", enabled=True),
        ToolkitConfig(class_name="FileToolkit", enabled=True),
    ]
)
```

Available built-in toolkits: `CalculatorToolkit`, `FileToolkit`, `WebSearchToolkit`, `E2BToolkit` (requires E2B API key), and more. See [TOOLKITS.md](TOOLKITS.md) for the full list.

---

## Next Steps

TODO

---

## Quick Reference

```python
from roma_dspy.core.engine.solve import solve
from roma_dspy.config.schemas.root import ROMAConfig
from roma_dspy.config.schemas.base import RuntimeConfig, LLMConfig
from roma_dspy.config.schemas.agents import AgentsConfig, AgentConfig

# Build config
cfg = AgentConfig(llm=LLMConfig(
    model="anthropic/claude-haiku-4-5-20251001",
    max_tokens=4000,
))
config = ROMAConfig(
    runtime=RuntimeConfig(timeout=700, max_depth=2),
    agents=AgentsConfig(
        atomizer=cfg, planner=cfg, executor=cfg, aggregator=cfg, verifier=cfg
    )
)

# Run a multi-step task
result = solve("Your complex multi-step task here", config=config)

# Access results
print(result.result)   # final synthesized answer
print(result.status)   # COMPLETED / FAILED
print(result.metrics)  # token usage, timing
