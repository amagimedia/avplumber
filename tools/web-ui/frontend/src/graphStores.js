import { writable } from 'svelte/store';

// Map<string, QueueStat>
// QueueStat is the object from queues.json: {name, capacity, occupied, pps, ...}
export const queueStatsByName = writable(new Map());


