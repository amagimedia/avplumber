// Station bug for the top-left corner: a mark inside a turning ring. Square window.
const graphic = Motion.graphic({
  html: '<div id="ring"></div><div id="mark">AVP</div>',
  css: `
    :host { display: grid; place-items: center; font-family: system-ui, sans-serif; }
    #ring { position: absolute; inset: 6%; border-radius: 50%; background: conic-gradient(#ff5a1f 0 30%, #ffffff40 30% 100%);
            -webkit-mask: radial-gradient(circle, transparent 60%, black 61%); mask: radial-gradient(circle, transparent 60%, black 61%); }
    #mark { position: relative; width: 62%; height: 62%; border-radius: 50%; background: #101c2bcc;
            display: grid; place-items: center; color: white; font-weight: 800; font-size: 26vh; }`,
  loop: [{ el: 'ring', spin: { seconds: 4 } }],
  demo: [[0, 'play']],
});
