// Record real UI controls and performance samples alongside host RTP capture.
const fs = require("fs"),
  { chromium } = require("playwright");
const root = process.argv[2] || ".",
  url = process.env.MIXER_UI_URL || "http://127.0.0.1:7681/";
fs.mkdirSync(root, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const b = await chromium.launch({
    channel: "chrome",
    args: ["--no-sandbox"],
  });
  const context = await b.newContext({
    viewport: { width: 1328, height: 734 },
    deviceScaleFactor: 2,
    ...(process.env.MIXER_HTTP_USER
      ? {
          httpCredentials: {
            username: process.env.MIXER_HTTP_USER,
            password: process.env.MIXER_HTTP_PASSWORD || "",
          },
        }
      : {}),
  });
  await context.addInitScript(() => {
    localStorage.setItem("avplumber.mixer.viewers", '["mv","program"]');
    localStorage.setItem("avplumber.mixer.tbarUpright", "true");
    localStorage.setItem("avplumber.mixer.filter", '"grid"');
  });
  await context.route("**/preview/**", (r) =>
    r.fulfill({
      contentType: "text/html",
      body: '<!doctype html><body style="background:black;margin:0">',
    }),
  );
  const p = await context.newPage();
  const get = async () => {
    const r = await context.request.get(url + "api/state");
    if (!r.ok()) throw Error("Show unavailable");
    return r.json();
  };
  const cmd = async (data) => {
    const r = await context.request.post(url + "api/command", { data });
    if (!r.ok()) throw Error(await r.text());
  };
  let original,
    changed = false;
  const report = { fps: 30, frames: 600, events: [] };
  try {
    await p.goto(url);
    await p.waitForFunction(() => state?.settings?.source_count === 192);
    original = await get();
    if (original.settings.canvas.fps !== 30) throw Error("Not the 30 fps show");
    if (
      original.status.lever.position !== 0 ||
      original.status.transition_manual
    )
      throw Error("T-bar is already in use");
    report.original = original;
    const pick = async (id) => {
      await p
        .locator(
          `#chips [data-focus="${id.match(/^(random|grid|pip|fullscreen|alpha_overlay)_/)[1]}"]`,
        )
        .click();
      for (let n = 0; n < 10; n++) {
        const tile = p.locator(`[data-scene="${id}"]`);
        if (await tile.count()) {
          await tile.click();
          return;
        }
        await p.locator("#pg-next").click();
      }
      throw Error(id);
    };
    const direct = async (on) => {
      const d = p.locator("#direct");
      if (((await d.getAttribute("aria-pressed")) === "true") !== on)
        await d.click();
    };
    changed = true;
    await direct(true);
    await pick("random_180");
    await sleep(500);
    for (const key of original.dsk)
      if (!key.on)
        await p.locator(`#keys button.take[data-focus="${key.id}"]`).click();
    await direct(false);
    await pick("pip_128");
    await p.locator("#fade").click();
    await sleep(5000);
    await p.evaluate(() => {
      window.captureTrace = [];
      window.captureStates = [];
      let last;
      window.captureTimer = setInterval(() => {
        const wall = (performance.timeOrigin + performance.now()) / 1000;
        captureTrace.push({
          wall,
          grip: gripShown,
          lever: {
            origin: lever.origin,
            position: lever.position,
            engaged: lever.engaged,
          },
          take: shownTake(),
          selected,
          direct,
          currentTransition,
          filter,
          scenePage,
        });
        if (state && state.status.now_pts_ms !== last) {
          last = state.status.now_pts_ms;
          captureStates.push({
            wall,
            state: JSON.parse(JSON.stringify(state)),
          });
        }
      }, 1000 / 30);
    });
    report.demo_start_wall = Date.now() / 1000;
    const start = Date.now();
    const at = async (sec) => {
      await sleep(Math.max(0, start + sec * 1000 - Date.now()));
      if ((await get()).setup_revision !== original.setup_revision)
        throw Error("Demo changed");
    };
    const stroke = async (from, to, duration) => {
      const t = await p.locator(".tbar .track").boundingBox(),
        g = await p.locator(".tbar .grip").boundingBox(),
        x = g.x + g.width / 2;
      await p.mouse.move(x, g.y + g.height / 2);
      await p.mouse.down();
      const began = Date.now();
      report.events.push({ wall: began / 1000, stroke: [from, to], duration });
      for (;;) {
        const f = Math.min(1, (Date.now() - began) / duration);
        await p.mouse.move(
          x,
          t.y + t.height - 18 - (from + (to - from) * f) * (t.height - 36),
        );
        if (f === 1) break;
        await sleep(30);
      }
      await p.mouse.up();
    };
    await at(2);
    await stroke(0, 0.5, 2000);
    await at(5);
    await stroke(0.5, 1, 2000);
    await at(9);
    await pick("random_180");
    await at(11);
    await stroke(1, 0, 4000);
    await at(17);
    await pick("random_179");
    await p.locator("#cut").click();
    await at(20);
    report.demo_end_wall = Date.now() / 1000;
    Object.assign(
      report,
      await p.evaluate(() => {
        clearInterval(captureTimer);
        return { trace: captureTrace, states: captureStates };
      }),
    );
    report.after = await get();
    report.ok = true;
    await sleep(2000);
  } finally {
    try {
      if (changed && (await get()).setup_revision === original.setup_revision) {
        await cmd({ command: "cut", scene: original.status.pgm_scene });
        await sleep(800);
        await cmd({ command: "preview", scene: original.status.pvw_scene });
        for (const key of original.dsk)
          await cmd({
            command: "dsk",
            key: key.id,
            on: key.on,
            fade_seconds: 0,
          });
      }
    } finally {
      fs.writeFileSync(root + "/capture.json", JSON.stringify(report));
      await b.close();
    }
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
