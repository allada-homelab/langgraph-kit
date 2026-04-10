# LLM Factory

**Source:** `src/langgraph_kit/llm.py`

The LLM factory creates a LangChain chat model from the package configuration, auto-detecting the provider from the model name.

## API

### build_llm()

```python
def build_llm() -> Any
```

Builds and returns a chat model based on the current `AgentConfig`. The return type is `Any` so callers don't need provider-specific stubs installed.

## Provider Detection

The model name prefix determines which LangChain class is instantiated:

| Prefix | Provider | Class | Extra Required |
|--------|----------|-------|----------------|
| `claude-*` | Anthropic | `ChatAnthropic` | `langgraph-kit[anthropic]` |
| `gemini-*` | Google | `ChatGoogleGenerativeAI` | `langgraph-kit[google]` |
| _(default)_ | OpenAI | `ChatOpenAI` | _(none)_ |

## Configuration Mapping

### OpenAI-compatible (default)

```python
ChatOpenAI(
    model=config.llm_model,
    streaming=True,
    base_url=config.llm_base_url,    # if non-empty
    api_key=config.llm_api_key,      # if non-empty
)
```

Supports any OpenAI-compatible API (OpenAI, Azure, vLLM, Ollama, LiteLLM, etc.) by setting `llm_base_url`.

### Anthropic

```python
ChatAnthropic(
    model=config.llm_model,
    streaming=True,
    api_key=config.llm_api_key,      # if non-empty
    base_url=config.llm_base_url,    # if non-empty
)
```

### Google

```python
ChatGoogleGenerativeAI(
    model=config.llm_model,
    google_api_key=config.llm_api_key,  # if non-empty
)
```

## Lazy Imports

Provider-specific packages are imported lazily inside their builder functions. This means you only need the extra installed for the provider you actually use — importing the module won't fail if `langchain-anthropic` isn't installed as long as you don't use a `claude-*` model.

## Example

```python
from langgraph_kit import AgentConfig, configure, build_llm

# OpenAI
configure(AgentConfig(llm_model="gpt-4o", llm_api_key="sk-..."))
llm = build_llm()  # ChatOpenAI

# Anthropic
configure(AgentConfig(llm_model="claude-sonnet-4-20250514", llm_api_key="sk-ant-..."))
llm = build_llm()  # ChatAnthropic

# Local vLLM
configure(AgentConfig(llm_model="qwen-2.5", llm_base_url="http://localhost:8000/v1"))
llm = build_llm()  # ChatOpenAI with custom base_url
```
