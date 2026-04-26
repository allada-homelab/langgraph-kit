# Live graph overlay

Pair `print_graph` (static structure) with the kit's SSE stream
(execution events) to highlight the currently-running node in real time.
This page is a recipe — the kit doesn't ship a hosted viewer; copy the
HTML below into your app or docs site and adapt it.

## Backend: render once, stream forever

```python
from langgraph_kit.core.visualization import print_graph
from langgraph_kit.streaming import stream_agent_events

# 1. Render the static diagram once at startup; serve it as part of
#    your viewer page.
mermaid_markup = print_graph(my_compiled_graph)

# 2. Drive runs through the kit's SSE streamer as usual. The stream
#    now also carries ``node_entered`` / ``node_exited`` events
#    whenever a graph-declared node enters or exits.
async for sse_chunk in stream_agent_events(my_compiled_graph, input_data, config):
    yield sse_chunk
```

The new event shape:

```json
data: {"node_entered": {"id": "<langgraph-run-id>", "name": "alpha"}}

data: {"node_exited":  {"id": "<langgraph-run-id>", "name": "alpha"}}
```

The `name` field matches the node id `print_graph` puts in the Mermaid
markup — same string in both places, so the viewer can use the name as
a CSS selector key. Internal kit machinery (memory extraction,
consolidation, prompt-injection scanner, etc.) is filtered out via the
existing `INTERNAL_TAG` convention; only nodes you declared on your
`StateGraph` reach the stream. Repeated `on_chain_start` events for the
same node (LangGraph fires extras when sub-channels fan in) are
coalesced — one `node_entered` per transition.

## Sample HTML/JS viewer

A self-contained 50-line page that renders the diagram and toggles an
`.active` CSS class on whichever node is currently executing. Drop into
a route on your app or paste into an `.html` file next to the kit's
docs site.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>langgraph-kit live overlay</title>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({ startOnLoad: false, theme: "neutral" });
  </script>
  <style>
    /* Neutral baseline so the toggle stands out. */
    .node rect, .node polygon, .node circle { fill: #f2f0ff; }
    /* The live class — added/removed by the script below as
       node_entered / node_exited events arrive. Tweak to taste. */
    .node.active rect,
    .node.active polygon,
    .node.active circle {
      fill: #4f46e5 !important;
      stroke: #312e81;
      stroke-width: 2px;
    }
  </style>
</head>
<body>
  <pre class="mermaid" id="diagram">
    /* Replace this with the markup from print_graph(graph),
       served by your app at e.g. /graph/<agent-id>/diagram.mmd. */
  </pre>

  <script type="module">
    // 1. Render the diagram.
    const { default: mermaid } = await import(
      "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs"
    );
    await mermaid.run({ querySelector: "#diagram" });

    // 2. Subscribe to the SSE stream. Replace the URL with your app's
    //    streaming endpoint (langgraph_kit.contrib.fastapi mounts one
    //    at POST /agents/{agent_id}/stream).
    const stream = new EventSource("/agents/my-agent/stream");

    function nodeEl(name) {
      // Mermaid generates ids like ``flowchart-alpha-N``; match by suffix.
      return [...document.querySelectorAll("g.node")].find(
        (el) => el.id.includes(`-${name}-`),
      );
    }

    let activeNode = null;

    stream.onmessage = (msg) => {
      // The kit emits one JSON payload per SSE message body. Parse and
      // dispatch on the kit's event keys.
      let payload;
      try { payload = JSON.parse(msg.data); }
      catch { return; }

      if (payload.node_entered) {
        activeNode?.classList.remove("active");
        activeNode = nodeEl(payload.node_entered.name);
        activeNode?.classList.add("active");
      } else if (payload.node_exited) {
        if (activeNode && activeNode.id.includes(`-${payload.node_exited.name}-`)) {
          activeNode.classList.remove("active");
          activeNode = null;
        }
      }
      // Ignore other event keys (token, tool_call_start, heartbeat,
      // etc.) — those are for the chat-side renderer.
    };
  </script>
</body>
</html>
```

## Caveats

* **Node id stability.** The overlay assumes `print_graph`'s Mermaid ids
  match the SSE event names. They do for nodes you declare on a
  `StateGraph` directly. Compiled subgraphs introduced via
  `expand_subgraphs=True` need their nodes addressed by their nested
  ids — adapt the `nodeEl` selector to your graph's structure.
* **Backpressure.** The kit coalesces same-node `node_entered` events
  but doesn't throttle distinct-node transitions. For a graph that
  fires 100+ transitions per second the viewer will queue updates;
  add your own debounce if that's a real workload.
* **Reconnection.** The kit emits SSE `id:` lines on every chunk. A
  client that reconnects with `Last-Event-ID` will (eventually) get a
  durable replay — the replay log is a follow-up. For now an
  interrupted overlay just resets state and resumes from whatever the
  next live event happens to be.
