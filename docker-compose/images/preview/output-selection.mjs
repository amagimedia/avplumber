// Mountpoints identify outputs; HEVC alone does not identify an HDR program.
export function playerOutputs(settings = {}, query = new URLSearchParams(), monitorMixer = true) {
  const explicit = Number(query.get("mountpoint")) || 0;
  const requested = query.get("codec") === "h265" ? "h265" : "h264";
  const programs = settings.program_outputs || [];
  const label = output => `${(output.color || "sdr").toUpperCase()} · ${output.codec === "h265" ? "H.265" : "H.264"}`;
  if (explicit) {
    const output = [...programs, ...(settings.preview_outputs || [])].find(o => o.mountpoint === explicit);
    const selected = {key: String(explicit), mountpoint: explicit, codec: output?.codec || requested,
      label: output ? label(output) : requested.toUpperCase()};
    return {choices: [selected], selected};
  }
  let choices;
  if (programs.length) {
    choices = programs.map(o => ({key: o.rendition, mountpoint: o.mountpoint, codec: o.codec, label: label(o)}));
  } else {
    const declared = query.get("outputs")?.split(",").filter(c => c === "h264" || c === "h265");
    const codecs = settings.preview_codecs?.length ? settings.preview_codecs :
      declared?.length ? declared : monitorMixer ? ["h264"] : ["h264", "h265"];
    choices = codecs.map(codec => ({key: codec, mountpoint: codec === "h265" ? 2 : 1,
      codec, label: codec === "h265" ? "H.265" : "H.264"}));
  }
  return {choices, selected: choices.find(o => o.key === query.get("rendition")) ||
    choices.find(o => o.codec === requested) || choices[0]};
}
