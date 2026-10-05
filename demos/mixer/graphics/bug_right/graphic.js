// Top-right bug: an on-air dot that blinks once a second and a wall clock. The window is twice as
// wide as high.
const graphic = Motion.graphic({
  html: '<div id="box"><div id="dot"></div><div id="clock"></div></div>',
  css: `
    :host { display: flex; align-items: center; justify-content: flex-end; font-family: system-ui, sans-serif; }
    #box { display: flex; align-items: center; gap: 6vh; padding: 0 8vh; height: 46%;
           border-radius: 23vh; background: #101c2bcc; color: white; }
    #dot { width: 16vh; height: 16vh; border-radius: 50%; background: #ff2b2b; }
    #clock { font: 700 22vh/1 ui-monospace, monospace; }`,
  // n counts half seconds of the wall clock: bright on the second, dim on the half. The clock is
  // written in the same call, so dot and text change in one painted frame, twice a second.
  every: [{
    seconds: 0.5,
    run: (el, n) => {
      el.dot.style.opacity = n % 2 ? '0.25' : '1';
      el.clock.textContent = new Date(n * 500).toTimeString().slice(0, 8);
    },
  }],
  demo: [[0, 'play']],
});
