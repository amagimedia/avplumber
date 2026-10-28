// Replay host pictures and captured controls in the matching real UI at Retina scale.
const fs = require("fs"),
  http = require("http");
const { chromium } = require("playwright");
const root = process.cwd(),
  capture = JSON.parse(fs.readFileSync(root + "/capture.json"));
const latest = (a, t) => a.findLast((s) => s.wall <= t) || a[0];
const report = JSON.parse(fs.readFileSync(root + "/report.json"));
let current = capture.states[0].state;
const server = http.createServer((req, res) => {
  let data,
    type = "text/html";
  if (req.url === "/") data = fs.readFileSync(root + "/ui.html");
  else if (req.url === "/outputs.js") {
    data = fs.readFileSync(root + "/outputs.js");
    type = "text/javascript";
  } else if (req.url === "/api/state") {
    data = JSON.stringify(current);
    type = "application/json";
  } else if (/^\/frames\/(pgm|preview)\/\d+\.png$/.test(req.url)) {
    data = fs.readFileSync(root + req.url);
    type = "image/png";
  } else data = '<!doctype html><body style="margin:0;background:black">';
  res.writeHead(200, { "Content-Type": type });
  res.end(data);
});
(async () => {
  await new Promise((r) => server.listen(48774, "127.0.0.1", r));
  const browser = process.env.CHROME_CDP_URL
    ? await chromium.connectOverCDP(process.env.CHROME_CDP_URL)
    : await chromium.launch({ channel: "chrome" });
  const context = await browser.newContext({
    viewport: { width: 1328, height: 734 },
    deviceScaleFactor: 2,
  });
  await context.addInitScript(() => {
    localStorage.setItem("avplumber.mixer.viewers", '["mv","program"]');
    localStorage.setItem("avplumber.mixer.tbarUpright", "true");
    localStorage.setItem("avplumber.mixer.filter", '"grid"');
    window.setInterval = () => 0;
  });
  const page = await context.newPage();
  page.on("pageerror", (e) => console.error(e));
  await page.goto("http://127.0.0.1:48774/");
  await page.waitForFunction(
    () => state?.settings?.source_count === 192 && viewers.length === 2,
  );
  await page.evaluate(() => {
    requestAnimationFrame = () => 0;
    refresh = async () => {};
    directTouched = tookTransition = true;
    turnTbar(true);
    selectTransition("fade");
    for (const v of viewers) {
      v.frame.style.display = "none";
      v.img = document.createElement("img");
      v.img.style.cssText =
        "position:absolute;inset:0;width:100%;height:100%;object-fit:fill";
      v.fit.insertBefore(v.img, v.ovl);
      v.statsRow.innerHTML = "<span>HOST CAPTURE</span><span>30 FPS</span>";
    }
  });
  fs.mkdirSync(root + "/ui-retina", { recursive: true });
  let geometry;
  const cdp = await context.newCDPSession(page);
  for (let n = 0; n < capture.frames; n++) {
    const wall = report.pgm.first_frame_wall + n / capture.fps,
      trace = latest(capture.trace, wall);
    current = latest(capture.states, wall).state;
    const info = await page.evaluate(
      async ({ snapshot, trace, n }) => {
        state = snapshot;
        selected = trace.selected;
        setDirect(trace.direct);
        filter = trace.filter;
        scenePage = trace.scenePage;
        currentTransition = trace.currentTransition;
        mix = gripBack = parking = null;
        lever.kind = "crossfade";
        lever.sync(trace.lever, true);
        render();
        renderTransition();
        renderLever();
        gripShown = trace.grip;
        document
          .querySelector(".tbar .track")
          .style.setProperty("--v", gripShown);
        for (const v of viewers) {
          v.frame.style.display = "none";
          v.statsRow.innerHTML = "<span>HOST CAPTURE</span><span>30 FPS</span>";
        }
        for (const [i, v] of viewers.entries()) {
          v.img.src =
            "/frames/" +
            (i === 0 ? "preview" : "pgm") +
            "/" +
            String(n + 1).padStart(3, "0") +
            ".png";
        }
        await Promise.all(viewers.map((v) => v.img.decode()));
        return viewers.map((v) => {
          const r = v.fit.getBoundingClientRect();
          return { x: r.x * 2, y: r.y * 2, w: r.width * 2, h: r.height * 2 };
        });
      },
      { snapshot: current, trace, n },
    );
    if (n === 0) {
      geometry = info;
      fs.writeFileSync(root + "/geometry.json", JSON.stringify(info));
      console.log("Geometry", info);
    }
    const shot = await cdp.send("Page.captureScreenshot", {
      format: "jpeg",
      quality: 95,
      fromSurface: true,
      captureBeyondViewport: false,
      clip: { x: 0, y: 0, width: 1328, height: 734, scale: 2 },
    });
    fs.writeFileSync(
      root + "/ui-retina/" + String(n).padStart(3, "0") + ".jpg",
      Buffer.from(shot.data, "base64"),
    );
    if (n % 60 === 0) console.log("Rendered", n);
    if (n === 140)
      await page.screenshot({ path: root + "/ui-retina-hold.png" });
  }
  await context.close();
  await browser.close();
  server.close();
  console.log("Rendered", capture.frames);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
