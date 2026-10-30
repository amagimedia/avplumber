the backend should be written in node.js and be minimal, i.e. contain minimal amount of logic, only websocket<->TCP converter and storing statistics history (in RAM, persistence not needed).

the web UI is for visualization, not remote take over of an instance: it may only query state. The backend forwards only the read-only queries listed in backend/requestGuards.js; never add a command that changes, starts, stops, adds or deletes anything, and never add a UI path that sends one.

the frontend should NOT be written in react or other heavyweight framework that causes older devices to render non-smoothly, but using lightweight framework/libraries like Svelte is allowed.

see @README.md for text API.

you are allowed to change c++ sources if needed for new features.