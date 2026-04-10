# HITL Models

**Source:** `src/langgraph_kit/core/hitl/models.py`

## ActionRequest

```python
class ActionRequest(BaseModel):
    action: str          # What the agent wants to do (e.g., "delete_file")
    args: dict = {}      # Action arguments (e.g., {"path": "config.yaml"})
```

## HumanInterruptConfig

```python
class HumanInterruptConfig(BaseModel):
    allow_ignore: bool = True    # User can dismiss without action
    allow_respond: bool = True   # User can type a response
    allow_edit: bool = True      # User can modify the action args
    allow_accept: bool = True    # User can approve as-is
```

## HumanInterrupt

The payload sent to the frontend when the agent pauses:

```python
class HumanInterrupt(BaseModel):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str             # Human-readable explanation
```

## HumanResponse

The user's response to an interrupt:

```python
class HumanResponse(BaseModel):
    type: str     # "accept", "ignore", "response", "edit"
    args: dict = {}  # Response data (e.g., edited args, text response)
```

## ResumeRequest

Request body for the resume endpoint:

```python
class ResumeRequest(BaseModel):
    responses: list[HumanResponse]  # One response per interrupt
```

## ThreadStateResponse

Response from the thread state endpoint:

```python
class ThreadStateResponse(BaseModel):
    thread_id: str
    status: str                    # "idle", "interrupted", "running"
    interrupts: list[HumanInterrupt] = []
```
