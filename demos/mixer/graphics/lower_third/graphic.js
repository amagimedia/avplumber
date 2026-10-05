// Name plate. The window is the graphic's own rectangle, so every size is in vh or % of that
// strip; outside the plate the window is transparent.
const PEOPLE = [
  { name: 'Ada Lovelace', role: 'Analyst · Engine No. 1' }, { name: 'Grace Hopper', role: 'Compiler desk' },
  { name: 'Alan Turing', role: 'Codebreaking correspondent' }, { name: 'Hedy Lamarr', role: 'Spread spectrum, live' }];
const graphic = Motion.graphic({
  html: '<div id="plate"><div id="accent"></div><div id="text"><div id="name"></div><div id="role"></div></div></div>',
  css: `
    :host { color: white; font-family: system-ui, sans-serif; }
    #plate { position: absolute; left: 0; top: 12%; width: 100%; height: 76%; display: flex; align-items: stretch; }
    #accent { width: 2.2%; background: #ff5a1f; }
    #text { flex: 1; background: linear-gradient(90deg, #101c2bee 0 70%, #101c2b00); padding: 0 4%;
            display: flex; flex-direction: column; justify-content: center; }
    #name { font-size: 30vh; font-weight: 700; line-height: 1.1; white-space: nowrap; }
    #role { font-size: 17vh; opacity: .85; white-space: nowrap; }`,
  data: PEOPLE[0],
  update: (el, data) => { el.name.textContent = data.name; el.role.textContent = data.role; },
  play: [{ el: 'plate', x: ['-104%', 0], opacity: [0, 1], seconds: 0.6, ease: [0.2, 0.8, 0.2, 1] }],
  stop: [{ el: 'plate', x: [0, '-104%'], opacity: [1, 0], seconds: 0.4, ease: 'in' }],
  // Each name is on for 5.2 s of a 6 s slot.
  demo: [...PEOPLE.flatMap((person, i) => [[6 * i, 'update', person], [6 * i, 'play'], [6 * i + 5.2, 'stop']]),
    [6 * PEOPLE.length, 'repeat']],
});
