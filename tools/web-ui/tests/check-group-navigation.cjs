// Node 22+. Read-only browser regression; never sends mixer commands.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
(async () => {
  const cdp = process.env.CDP_URL || 'http://127.0.0.1:9222';
  const target = process.argv[2] || 'http://127.0.0.1:22222/';
  const page = await fetch(`${cdp}/json/new?${encodeURIComponent(target)}`, {method: 'PUT'}).then(r => r.json());
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise(resolve => ws.addEventListener('open', resolve, {once: true}));
  const pending = new Map(), errors = []; let id = 0;
  ws.addEventListener('message', event => {
    const m = JSON.parse(event.data), p = pending.get(m.id);
    if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails);
    if (p) { pending.delete(m.id); clearTimeout(p.timer); m.error ? p.reject(m.error) : p.resolve(m.result); }
  });
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    const request = ++id;
    pending.set(request, {resolve, reject, timer: setTimeout(() => {
      pending.delete(request); reject(new Error(method + ' timeout'));
    }, 15000)});
    ws.send(JSON.stringify({id: request, method, params}));
  });
  const evaluate = async expression => {
    const reply = await call('Runtime.evaluate', {expression, returnByValue: true});
    if (reply.exceptionDetails) throw new Error(JSON.stringify(reply.exceptionDetails));
    return reply.result.value;
  };
  async function wait(expression) {
    for (let i = 0; i < 200; ++i) { if (await evaluate(expression)) return; await delay(100); }
    throw new Error('UI condition failed: ' + expression);
  }
  const ready = '!document.querySelector(".graph-loading") && !document.querySelector(".rete-error")';
  const view = name => `document.querySelector('[data-graph-view="${name}"]')`;
  const clickTitle = title => evaluate(`[...document.querySelectorAll('[data-testid=node]')].find(n => n.innerText.split('\\n')[0] === ${JSON.stringify(title)}).click()`);
  async function inspectBoxes(label, minimumWidth = 0) {
    await delay(100);
    const boxes = await evaluate(`[...document.querySelectorAll('[data-testid=node]')].map(n => ({text:n.innerText,rect:n.getBoundingClientRect().toJSON()}))`);
    const canvas = await evaluate('document.querySelector(".rete-container").getBoundingClientRect().toJSON()');
    assert(boxes.length, label + ': empty graph');
    for (const box of boxes) {
      assert(box.rect.width >= minimumWidth, label + ': unreadably small (' + box.rect.width + 'px) ' + box.text);
      assert(box.rect.left >= canvas.left - 1 && box.rect.right <= canvas.right + 1 &&
        box.rect.top >= canvas.top - 1 && box.rect.bottom <= canvas.bottom + 1,
        label + ': node clipped outside graph viewport: ' + box.text);
    }
    for (let i=0;i<boxes.length;++i) for (let j=i+1;j<boxes.length;++j) {
      const a=boxes[i].rect,b=boxes[j].rect;
      assert(a.right<=b.left+1 || b.right<=a.left+1 || a.bottom<=b.top+1 || b.bottom<=a.top+1,
        label + ': overlapping ' + boxes[i].text + ' / ' + boxes[j].text);
    }
    const crossings = await evaluate(`(() => {
      const nodes = [...document.querySelectorAll('[data-testid=node]')].map(n => ({name:n.innerText.split('\\n')[0], r:n.getBoundingClientRect()}));
      const errors = [];
      for (const path of document.querySelectorAll('svg[data-testid=connection] > path')) {
        const d = path.getAttribute('d') || '';
        if (/[CQASTHVZ]/i.test(d)) {errors.push('non-orthogonal path: ' + d); continue;}
        const values = (d.match(/-?\\d*\\.?\\d+(?:e[-+]?\\d+)?/gi) || []).map(Number);
        const matrix = path.getScreenCTM();
        const points = [];
        for (let i=0;i<values.length;i+=2) points.push(new DOMPoint(values[i], values[i+1]).matrixTransform(matrix));
        for(let i=1;i<points.length;i++) {
          const a=points[i-1],b=points[i];
          for(const {name,r} of nodes) {
            const vertical=Math.abs(a.x-b.x)<1 && a.x>r.left+2 && a.x<r.right-2 && Math.max(a.y,b.y)>r.top+2 && Math.min(a.y,b.y)<r.bottom-2;
            const horizontal=Math.abs(a.y-b.y)<1 && a.y>r.top+2 && a.y<r.bottom-2 && Math.max(a.x,b.x)>r.left+2 && Math.min(a.x,b.x)<r.right-2;
            if(vertical || horizontal) errors.push('edge crosses ' + name);
          }
        }
      }
      return [...new Set(errors)];
    })()`);
    assert.deepEqual(crossings, [], label + ': connections cross node interiors');
    return boxes;
  }
  async function capture(name) {
    if (!process.env.SCREENSHOT_DIR) return;
    fs.mkdirSync(process.env.SCREENSHOT_DIR, {recursive:true});
    const shot=await call('Page.captureScreenshot',{format:'png'});
    fs.writeFileSync(path.join(process.env.SCREENSHOT_DIR, `${name}.png`), Buffer.from(shot.data,'base64'));
  }
  async function overview() {
    await evaluate('document.querySelector(".graph-overview").click()');
    await wait(`${view('overview')} && ${ready} && document.querySelectorAll('[data-testid=node]').length === ${initial}`);
  }
  let initial;
  try {
    await call('Runtime.enable');
    await call('Emulation.setDeviceMetricsOverride', {width:1920,height:1080,deviceScaleFactor:1,mobile:false});
    await wait(`${view('overview')} && ${ready} && document.querySelectorAll('[data-testid=node]').length > 1`);
    for (const [setting, label] of [['LIVE_QUEUE_STATS', 'live graph queue stats'], ['AUTO_REFRESH_QUEUES', 'auto-refresh queue fill']]) {
      if (!(setting in process.env)) continue;
      const enabled = process.env[setting] === '1';
      await evaluate(`(() => {const label = [...document.querySelectorAll('.toolbar label')].find(n => n.innerText.includes(${JSON.stringify(label)})); const input = label.querySelector('input'); if(input.checked !== ${enabled}) input.click();})()`);
    }
    const counts = await evaluate('({...document.querySelector(".graph-native-counts").dataset})');
    const nativeCount = Number(counts.nodes);
    const titles = await evaluate('[...document.querySelectorAll("[data-testid=node]")].map(n => n.innerText.split("\\n")[0])');
    initial = titles.length;
    assert.equal(await evaluate('!!document.querySelector(".graph-back")'), false, 'overview must not have a parent');
    await inspectBoxes('overview', 140);
    await evaluate('if (document.querySelector("[data-graph-view]").getBoundingClientRect().width < window.innerWidth * 0.9) document.querySelector(".graph-expand").click()');
    await delay(350);
    await wait('document.querySelector("[data-graph-view]").getBoundingClientRect().width > window.innerWidth * 0.9');
    await capture('overview');
    for (const title of titles) {
      await clickTitle(title);
      await wait(`(${view('family')} || ${view('group')}) && ${ready}`);
      assert(await evaluate('document.querySelector("[data-graph-view] .toolbar").firstElementChild.matches(".graph-back")'), 'Back must be the first toolbar control');
      if (await evaluate(`!!${view('family')}`)) {
        await capture('input-family');
        const members = await evaluate('[...document.querySelectorAll(".group-browser strong")].map(n => n.textContent)');
        for (const member of members) {
          await evaluate(`[...document.querySelectorAll('.group-browser button')].find(n=>n.querySelector('strong').textContent === ${JSON.stringify(member)}).click()`);
          await wait(`${view('group')} && ${ready} && document.querySelectorAll('[data-testid=node]').length > 1`);
          const boxes = await inspectBoxes(member, 140);
          assert(boxes.some(b => b.text.includes(member.replace('input_', 'decode_'))), member + ': missing decoder');
          if (member === members[0]) {
            const nativeName = member.replace('input_', 'decode_');
            await clickTitle(nativeName);
            await wait(`document.querySelector('.avp-selected')?.innerText.startsWith(${JSON.stringify(nativeName)})`);
            await clickTitle('External outputs');
            await wait(`document.querySelector('.avp-selected')?.innerText.startsWith(${JSON.stringify(nativeName)})`);
            await capture('input-chain');
          }
          await evaluate('document.querySelector(".graph-back").click()');
          await wait(`!!${view('family')}`);
          assert.equal(await evaluate('document.querySelectorAll(".group-browser button").length'), members.length, 'Back must return to the same family');
        }
      } else {
        await inspectBoxes(title, 140);
        await capture(title.replace(/[^a-z0-9_-]/gi, '_'));
        if (title === 'mixer') {
          await evaluate('document.querySelector(".graph-expand").click()');
          await delay(400);
          await wait(`${view('group')} && ${ready}`);
          await inspectBoxes('restored mixer group');
          await evaluate('document.querySelector(".graph-expand").click()');
          await delay(400);
          await wait(`${view('group')} && ${ready}`);
          await inspectBoxes('expanded mixer group', 140);
        }
      }
      await evaluate('document.querySelector(".graph-back").click()');
      await wait(`${view('overview')} && ${ready} && document.querySelectorAll('[data-testid=node]').length === ${initial}`);
      assert.equal(await evaluate('!!document.querySelector(".graph-back")'), false);
    }
    // Overview remains an explicit shortcut, separate from single-level Back.
    await clickTitle(titles.find(title => !title.endsWith('*')));
    await wait(`${view('group')} && ${ready}`);
    await overview();
    await clickTitle(titles.find(title => !title.endsWith('*')));
    await wait(`${view('group')} && ${ready}`);
    await evaluate('document.querySelector(".graph-grouped-toggle").click()');
    await wait(`${view('full')} && ${ready} && document.querySelectorAll('[data-testid=node]').length === ${nativeCount}`);
    await inspectBoxes('full graph');
    // Queue telemetry must update Svelte attributes/text without replacing Rete sockets.
    await evaluate('window.__graphTestNodes = [...document.querySelectorAll("[data-testid=node]")]; window.__graphTestSockets = [...document.querySelectorAll("[data-testid=socket]")]');
    await delay(2200);
    assert(await evaluate('window.__graphTestNodes.every(n => n.isConnected) && window.__graphTestSockets.every(n => n.isConnected)'), 'queue updates replaced nodes or sockets');
    await call('Emulation.setDeviceMetricsOverride', {width:Number(process.env.FULL_WIDTH || 6000),height:Number(process.env.FULL_HEIGHT || 4000),deviceScaleFactor:1,mobile:false});
    await evaluate('document.querySelector(".graph-grouped-toggle").click()');
    await wait(`${view('overview')} && ${ready} && document.querySelectorAll('[data-testid=node]').length === ${initial}`);
    await evaluate('document.querySelector(".graph-grouped-toggle").click()');
    await wait(`${view('full')} && ${ready} && document.querySelectorAll('[data-testid=node]').length === ${nativeCount}`);
    await inspectBoxes('full graph export', 140);
    await capture('full-graph');
    await call('Emulation.setDeviceMetricsOverride', {width:1920,height:1080,deviceScaleFactor:1,mobile:false});
    await evaluate('document.querySelector(".graph-grouped-toggle").click()');
    await wait(`${view('overview')} && ${ready} && document.querySelectorAll('[data-testid=node]').length === ${initial}`);
    for (let n=0;n<6;n++) {
      await evaluate('document.querySelector(".graph-grouped-toggle").click()');
      await delay(35);
    }
    await wait(`${view('overview')} && ${ready} && document.querySelectorAll('[data-testid=node]').length === ${initial}`);
    await inspectBoxes('overview after rapid toggles', 140);
    assert.deepEqual(await evaluate('({...document.querySelector(".graph-native-counts").dataset})'), counts);
    assert.deepEqual(errors, [], 'uncaught browser exceptions');
    console.log(`PASS: ${nativeCount} native nodes / ${counts.queues} defined queues / ${initial} blocks; all input/group views and single-level Back; native inspection; full graph; rapid mode switches; no overlaps/exceptions`);
  } catch (error) {
    await capture('failure');
    throw error;
  } finally {
    for (const request of pending.values()) clearTimeout(request.timer);
    ws.close();
    await fetch(`${cdp}/json/close/${page.id}`);
  }
})().catch(error => { console.error(error); process.exit(1); });
