const refs = (value) => typeof value === 'string' ? [value] : Array.isArray(value) ? value.filter(v => typeof v === 'string') : [];
export const sourceQueues = (params = {}) => refs(params.src);
export const destinationQueues = (params = {}) => [...new Set([
  ...refs(params.dst), ...Object.values(params.routing || {}).flatMap(refs),
])];

// A projection only: native nodes/queues remain available for inspection.
export function groupGraph(nodes = [], queues = [], expanded = new Set()) {
  const groups = new Set(nodes.map(n => n.params?.group).filter(Boolean));
  const families = new Map();
  for (const group of groups) {
    const family = group.replace(/\d+$/, '*');
    if (family !== group) families.set(family, [...(families.get(family) || []), group]);
  }
  const familyOf = new Map();
  for (const [family, members] of families) {
    if (members.length > 1) for (const group of members) familyOf.set(group, family);
  }
  const owners = new Map(), visible = new Map(), membership = new Map();
  for (const node of nodes) {
    const group = node.params?.group, family = familyOf.get(group);
    const collapsed = family && !expanded.has(`family:${family}`) ? `family:${family}`
      : group && !expanded.has(`group:${group}`) ? `group:${group}` : null;
    const key = collapsed ? `@view:${collapsed}` : node.name;
    owners.set(node.name, key);
    if (!visible.has(key)) {
      visible.set(key, { ...node, name: key, params: { src: [], dst: [] } });
      if (collapsed) membership.set(key, { key: collapsed, label: collapsed.slice(collapsed.indexOf(':') + 1), nodes: [], internal: [] });
    }
    if (collapsed) membership.get(key).nodes.push(node);
  }
  const defined = new Map();
  for (const node of nodes) {
    for (const [field, names] of [['producers', destinationQueues(node.params)], ['consumers', sourceQueues(node.params)]]) {
      for (const name of names) {
        if (!defined.has(name)) defined.set(name, { producers: new Set(), consumers: new Set() });
        defined.get(name)[field].add(owners.get(node.name));
      }
    }
  }
  const bundles = new Map();
  for (const [name, ends] of defined) {
    const producers = [...ends.producers].sort(), consumers = [...ends.consumers].sort();
    if (producers.length === 1 && consumers.length === 1 && producers[0] === consumers[0] && membership.has(producers[0])) {
      membership.get(producers[0]).internal.push(name);
      continue;
    }
    // Never combine individual diagnostic edges between fully expanded nodes.
    const collapsed = [...producers, ...consumers].some(key => membership.has(key));
    const key = collapsed ? JSON.stringify([producers, consumers]) : JSON.stringify([name]);
    if (!bundles.has(key)) bundles.set(key, { producers, consumers, names: [] });
    bundles.get(key).names.push(name);
  }
  const stats = new Map(queues.map(q => [q.name, q]));
  const projectedQueues = [];
  let index = 0;
  for (const bundle of bundles.values()) {
    const name = bundle.names.length === 1 ? bundle.names[0] : `${++index}: ${bundle.names.length} queues`;
    for (const owner of bundle.producers) visible.get(owner).params.dst.push(name);
    for (const owner of bundle.consumers) visible.get(owner).params.src.push(name);
    const samples = bundle.names.map(n => stats.get(n)).filter(Boolean);
    const sum = key => samples.reduce((total, q) => total + (Number(q[key]) || 0), 0);
    projectedQueues.push({ name, capacity: sum('capacity'), occupied: sum('occupied'), pps: sum('pps'),
      members: bundle.names, aggregate: bundle.names.length > 1 });
  }
  for (const [key, group] of membership) {
    Object.assign(visible.get(key), {
      label: group.label, type: `${group.nodes.length} nodes · ${group.internal.length} internal queues`,
      working: group.nodes.some(n => n.working),
    });
  }
  return { nodes: [...visible.values()], queues: projectedQueues, membership,
    nodeCount: nodes.length, queueCount: defined.size,
    internalCount: [...membership.values()].reduce((n, g) => n + g.internal.length, 0),
  };
}

export function focusGroup(nodes, queues, groupName) {
  const inside = nodes.filter(n => n.params?.group === groupName);
  const produced = new Set(inside.flatMap(n => destinationQueues(n.params)));
  const consumed = new Set(inside.flatMap(n => sourceQueues(n.params)));
  const incoming = [...consumed].filter(q => !produced.has(q));
  const outsideConsumed = new Set(nodes.filter(n => n.params?.group !== groupName).flatMap(n => sourceQueues(n.params)));
  const outgoing = [...produced].filter(q => outsideConsumed.has(q) || !consumed.has(q));
  const scoped = [...inside];
  if (incoming.length) scoped.push({name:'@boundary:in', type:'boundary', working:true,
    params:{group:'External inputs', dst:incoming}});
  if (outgoing.length) scoped.push({name:'@boundary:out', type:'boundary', working:true,
    params:{group:'External outputs', src:outgoing}});
  const view = groupGraph(scoped, queues, new Set([`group:${groupName}`]));
  for (const node of view.nodes) {
    if (view.membership.has(node.name)) node.type = 'Group boundary';
  }
  return view;
}
