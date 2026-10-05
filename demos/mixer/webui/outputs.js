// The outputs the mixer encodes and where the player plays each, shared by the mixer page (index.html)
// and the wall (wall.html). A classic script: its declarations are globals of the page that loads it.
const codecName = (codec) => ({h264: "H.264", h265: "H.265"})[codec] || String(codec || "").toUpperCase();
const canvasText = (c) => `${c.width}×${c.height}p${c.fps}`;
const formatText = (c) => c ? `${canvasText(c)} ${({nv12: "SDR 4:2:0", p010le: "HDR 4:2:0", p210le: "HDR 4:2:2"})[c.working_format] || "HDR"}` : "";

const query = new URLSearchParams(location.search);
// Where the player is, for every output's URL: the backend's --preview-base (a path on this origin,
// such as /preview/ behind a reverse proxy, or a URL), else port 8080 of this host. ?preview_port=
// replaces the port either way.
const config = JSON.parse(document.getElementById("config").textContent);
const playerBase = new URL(config.preview_base || "/", location.href);
const previewPort = query.get("preview_port") || (!config.preview_base && "8080");
if (previewPort) playerBase.port = previewPort;
playerBase.search = playerBase.hash = "";
// The player's own H.265 message sits in the footer the viewers clip, so the page never asks for a
// codec the browser cannot receive.
const h265Playable = (() => {
  try { return RTCRtpReceiver.getCapabilities("video").codecs.some((c) => c.mimeType.toLowerCase() === "video/h265"); }
  catch (_) { return false; }
})();
const playable = (codec) => codec !== "h265" || h265Playable;
// ?codec= picks the program's default codec; HDR otherwise, where the browser can play it.
const defaultCodec = (codecs) => [query.get("codec"), "h265", "h264"].find((c) => codecs.includes(c) && playable(c)) || codecs[0];
// The dirty program plays from the player's own mountpoints in the preview codecs; every other output
// is an H.264 mountpoint of its own, which the player plays as H.264 whatever codec the URL names.
// Every viewer, the program and the multiviews alike, plays with the same receiver playout delay:
// the browser's adaptive jitter buffer, or with ?lowlat=1 on this page the player's own ?lowlat=1
// for all of them, the buffer pinned to its minimum (jitterBufferTarget and playoutDelayHint 0), so
// a cut shows in the program and in the multiview's PVW tile at the same instant instead of each
// stream settling on its own delay. Opt-in, as on the player: the pin gains 4-5 ms on a clean link
// and removes the cushion on a bad one, and a software HEVC decoder drops late frames, while the
// program viewport defaults to H.265 wherever the browser plays it.
const lowLatency = query.get("lowlat") === "1";
// `codecs`: the program renditions the player offers, the mixer's preview_codecs.
function playerUrl(output, codec, codecs) {
  const url = new URL(playerBase);
  if (output) url.searchParams.set("mountpoint", output.mountpoint);
  else {
    if (codecs?.length) url.searchParams.set("outputs", codecs.join(","));
    url.searchParams.set("codec", codec);
  }
  if (lowLatency) url.searchParams.set("lowlat", "1");
  return url.href;
}
// Dirty renditions publish only their codec; like the player's own picker, H.265 is the HDR one.
// On an SDR canvas both are SDR, so the codec names them.
const RANGE_OF_CODEC = {h264: "SDR", h265: "HDR"};
function programOptions(s, keys) {
  const codecs = s.preview_codecs?.length ? s.preview_codecs : ["h265", "h264"].filter(playable);
  const preferred = defaultCodec(codecs), name = keys ? "Program dirty" : "Program";
  // The player shows its codec picker when the mixer has two program codecs.
  const base = {group: "Program", dot: "var(--pgm)", program: true, chrome: s.preview_codecs?.length > 1 ? 60 : 49};
  // Without the mixer's codecs yet, or with one, a single entry keeps the value saved by older pages.
  if (!s.preview_codecs?.length || codecs.length < 2) {
    return [{...base, value: "program", label: name, preferred: true, url: playerUrl(null, preferred, s.preview_codecs),
             unplayable: !playable(preferred),
             meta: [keys && "with keys", s.preview_codecs?.length && codecName(preferred), formatText(s.canvas)].filter(Boolean).join(" · ")}];
  }
  const hdr = s.canvas.working_format !== "nv12";
  return codecs.filter(playable).map((codec) => ({
    ...base, value: `program:${codec}`, label: `${name} · ${(hdr && RANGE_OF_CODEC[codec]) || codecName(codec)}`,
    preferred: codec === preferred, url: playerUrl(null, codec, s.preview_codecs),
    meta: [keys && "with keys", codecName(codec), s.canvas && canvasText(s.canvas)].filter(Boolean).join(" · "),
  }));
}
// A preview output, the clean feed or an aux bus: an H.264 or H.265 mountpoint of its own.
const previewOption = (o) => ({value: o.bus, label: o.label || o.bus, url: playerUrl(o), chrome: 49, unplayable: !playable(o.codec)});
// Canvas pixels of an output: an aux bus reports its own, the program has the mixer's.
function canvasSize(s, aux) {
  const c = s.canvas;
  return aux?.canvas ? [aux.canvas.w, aux.canvas.h] : c ? [c.width, c.height] : [9, 16];
}
