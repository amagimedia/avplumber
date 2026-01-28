the backend should be written in node.js and be minimal, i.e. contain minimal amount of logic, only websocket<->TCP converter and storing statistics history (in RAM, persistence not needed).

the frontend should NOT be written in react or other heavyweight framework that causes older devices to render non-smoothly, but using lightweight framework/libraries like Svelte is allowed.

see @README.md for text API.

you are allowed to change c++ sources if needed for new features.