// Codec and color routing shared by /wall, the mixer and the standalone player.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {playerOutputs} from "../../../docker-compose/images/preview/output-selection.mjs";

const settings = {
  canvas: {width: 1920, height: 1080, fps: 60, working_format: "p010le"},
  preview_codecs: ["h265"],
  program_outputs: [
    {rendition: "sdr", mountpoint: 1, codec: "h265", color: "sdr", fps: 60},
    {rendition: "hdr", mountpoint: 2, codec: "h265", color: "hdr", fps: 60},
  ],
  preview_outputs: [{bus: "mv", mountpoint: 5008, codec: "h265", color: "sdr"}],
};
const source = fs.readFileSync(new URL("../webui/outputs.js", import.meta.url), "utf8");
function page(hevc) {
  const context = vm.createContext({URL, URLSearchParams,
    location: {href: "http://127.0.0.1/mixer/", search: ""},
    document: {getElementById: () => ({textContent: '{"preview_base":"/preview/"}'})},
    RTCRtpReceiver: {getCapabilities: () => ({codecs: hevc ? [{mimeType: "video/H265"}] : []})}, settings});
  vm.runInContext(source, context);
  return expression => JSON.parse(JSON.stringify(vm.runInContext(expression, context)));
}
const evaluate = page(true);
const programs = evaluate("programOptions(settings, false)");
assert.equal(programs.length, 2);
assert.deepEqual(programs.map(o => o.value), ["program:sdr", "program:hdr"]);
assert.match(programs[0].label, /SDR.*H\.265/);
assert.match(programs[1].label, /HDR.*H\.265/);
assert.equal(programs[1].preferred, true);
for (const [index, output] of programs.entries()) {
  const query = new URL(output.url).searchParams;
  assert.equal(query.get("mountpoint"), String(index + 1));
  assert.equal(query.get("codec"), "h265");
  assert.equal(query.get("color"), index ? "hdr" : "sdr");
}
assert.equal(new URL(evaluate("previewOption(settings.preview_outputs[0])").url).searchParams.get("codec"), "h265");
assert.ok(page(false)("programOptions(settings, false)").every(o => o.unplayable));
assert.equal(evaluate("programOptions({preview_codecs:['h264'], canvas:settings.canvas}, false)")[0].value, "program");

let player = playerOutputs(settings, new URLSearchParams("codec=h265&rendition=hdr"));
assert.equal(player.choices.length, 2);
assert.equal(player.selected.mountpoint, 2);
player = playerOutputs(settings, new URLSearchParams("codec=h265&rendition=sdr"));
assert.equal(player.selected.mountpoint, 1);
player = playerOutputs(settings, new URLSearchParams("mountpoint=5008&codec=h264"));
assert.equal(player.choices.length, 1);
assert.equal(player.selected.codec, "h265"); // current contract overrides a stale iframe URL
assert.equal(player.selected.mountpoint, 5008);
player = playerOutputs(undefined, new URLSearchParams("mountpoint=5020&codec=h265"));
assert.equal(player.selected.codec, "h265");
assert.equal(playerOutputs({preview_codecs: ["h264"]}).selected.mountpoint, 1);
assert.equal(playerOutputs({}, new URLSearchParams("codec=h265"), false).selected.mountpoint, 2);
console.log("Output routing: SDR/HDR HEVC, AUX HEVC, unavailable HEVC and legacy H.264 passed");
