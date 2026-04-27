import { writable } from 'svelte/store';

// Map<string, QueueStat>
// QueueStat is the object from queues.json: {name, capacity, occupied, pps, ...}
export const queueStatsByName = writable(new Map());

/** Queue name selected from graph or queues table (empty = none). */
export const selectedQueueName = writable('');

/** Queue under the pointer on a graph edge (for list + edge highlight). */
export const graphHoveredQueueName = writable('');


