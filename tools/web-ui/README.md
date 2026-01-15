# avplumber web-ui

A web UI made in Node & Svelte, for viewing:
* the graph, with animated averaged packet flow in queues
* queues statistics (fill, pps)
* node creation-time parameters
* node objects (`node.object.get`)
* statistics from instance-shared objects (realtime, sentinel)
* stream analysis statistics from `stats.subscribe`

## How to use

```
cd frontend
npm run build
npm start
```

and when starting avplumber, specify `--webui-api` and `--instance-name`. Also, make sure that `--port` is specified, otherwise web UI won't be able to execute necessary commands.

## DiSCLAiMER

I'm not a frontend developer.

Written by AI.

