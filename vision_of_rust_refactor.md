# Vision of Rust Refactor

* control commands must be backward-compatible
* node authoring in Rust must be as simple as it is now in C++ (preferably even simpler, see boilerplate in c++ non-blocking nodes)
* adapters and C ABI necessary for running existing C++ nodes in the new engine will be ugly anyway so do them best-effort, and ABSOLUTELY DO NOT SACRIFICE NATIVE RUST API/ENGINE CLEANINESS FOR THIS
* pyplumber - ???

Make this port follow the spirit of the original C++ avplumber. It needs to be simple. If I wanted something complicated, I would go with GStreamer. CORPO-CODE PROHIBITED. This started as a personal project in a startup, let's keep it like this.


## Node authoring API

Node author writes:

* JSON object -> node parameters struct converter - optional - node constructor may also use raw JSON objects
* node constructor, destructor if needed
* node body

### Node logic implementation:

Grain processing logic must have clean ENTRY and EXIT points.

Possible paradigms:

* as in C++ blocking nodes - explicit dequeue & enqueue, simple looped function, requires separate thread - thread ending mumbo-jumbo (process() may be waiting for packet that will never come)
* as in C++ non-blocking nodes - explicitly requesting wakeup from the EventLoop - complicates the node code but useful for maximum control / atypical cases
* node author just writes the grain transform function that transforms grain into 0, 1 or multiple grains - requires special cases (e.g. timeout) for atypical nodes like the Sentinel, but otherwise VERY GOOD (straighforward grain flow) for nodes with a single input pad. This API is special in a way - it can be shared between threaded and async execution engines (and it is user's responsibility to choose right). One more "conceptual edge case" - if the node wants to consume >1 grain and emit 1, it needs to explicitly maintain internal state (on struct etc), not sure how ugly it would be in practice (already sort of - in force_fps.cpp since it became non-blocking node). In case of nodes with multiple input pads, similar buffering is also likely to happen, unless grain is outputted immediately.

We're using Rust so we have async/await, which brings us to another that wasn't possible in C++:

* node body is written as async function, awaited function can be wrapped in timeout or `select!` - feels native!


### Other convenience considerations

Node may request graceful reboot of itself from the engine (without usual auto_restart action) or tell the engine beforehand to reboot the node on every input format change

No multi-layer node creation please. Is it necessary at all?! Convert JSON to node-specific parameters and that's what is needed to create the node. If node instance is input-format-specific, delay the creation until it is available, and recreate on change.


## Execution engines

In C++ avplumber there was no explicit concept of execution engine - all were running simultaneously - nodes in their own threads and nodes in the event loops. It was this straightforward and possible because all queues were based on eventfd, no matter the node (blocking/non-blocking).

Now we want direct edges - edges without queues (not even 1-element implicit on coroutine stack) so what???


### Idea - chains

Pushing a grain into the edge makes the next node wake up and consume it immediately. Wait - "wake up"? It is old concept from thread or event loop. Can't nodes react to grains directly? Decoupled in API, so the nodes author doesn't know what comes next, but coupled in the runtime, by the framework? So no wake up (unless the node has its own thread, then it is literally a wakeup), just reaction to event, callback or `impl fn` called.

Chain = grain-transform nodes in a row.

Additional API: is space free? 1:1 nodes short-circuit this (space in input pad always equals space in output pad). Useful for, e.g. fps conversions that may duplicate or drop. So this API needs optional timestamp argument. But it makes timestamp a special citizen. But is it bad? There are no other candidates for special citizens in this case I suppose.

What do we need this direct edges for: to make lightweight nodes behave as a list of instructions to do with the grain. No explicit or implicit buffering between lightweight nodes, because it contributes to latency. So it is OK to have buffering if needed for the algorithm itself (e.g. `force_fps`), but not between nodes.

In all cases, must prevent stack growth in complicated multi-layer graphs. So engine must manage it somehow, no direct calls between nodes.


## The whole idea crystallized

Support several node APIs:
* threaded (old blocking) - good for libavformat input that is the packet source (the flow starts from it) but uses blocking functions so has to be threaded
* async (old non-blocking) - good for e.g. sentinel that needs to output frames on its own. Ditch the invented here event loop - use Rust native futures. They can be used in `async { ...().await }` or handled manually - whatever the node author prefers.
* grain transform - good for e.g. filtering, conversions, reinterpret_planes, resampler etc. Node author implements only callbacks or functions in a trait, so a single code can be both blocking and non-blocking (decided in runtime)


