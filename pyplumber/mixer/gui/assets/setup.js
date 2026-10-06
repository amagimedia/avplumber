/* Shared setup widgets; applications supply canvas choices, output names and NVENC limits. */
window.MixerSetup = {
 canvasControls(container, choices, selected={}) {
  const labels={orientation:'Orientation',fps:'Frames per second',mode:'Mode'};
  for(const [id,values] of Object.entries(choices)) {
   const label=document.createElement('label'),select=document.createElement('select');
   label.textContent=labels[id]??id;select.id=id;
   select.append(...values.map(v=>new Option(...(Array.isArray(v)?[v[1],v[0]]:[String(v),String(v)]))));
   if(selected[id]!==undefined)select.value=String(selected[id]);
   label.append(select);container.append(label);
  }
 },
 encodeRow({output:o,encode,nvenc,onChange,codecs=['h264_nvenc','hevc_nvenc'],count=null}) {
  const row=document.createElement('div'),[low,high]=nvenc.bitrate_kbps;
  row.className='source output';
  row.innerHTML='<div><b></b><small></small></div><span class="count"></span>'+
   `<select class="codec"></select><select class="preset"></select><input class="mbit" type="number" min="${low/1000}" max="${high/1000}" step="0.05">`;
  const codec=row.querySelector('.codec'),preset=row.querySelector('.preset'),mbit=row.querySelector('.mbit');
  codec.append(...codecs.map(value=>new Option(value==='hevc_nvenc'?'H.265':'H.264',value)));
  codec.value=encode.codec;codec.setAttribute('aria-label',`${o.name} codec`);
  codec.oninput=()=>{encode.codec=codec.value;onChange();};
  preset.append(...Object.keys(nvenc.pct_per_fps[o.codec]).map(value=>new Option(value)));
  preset.value=encode.preset;mbit.value=encode.bitrate_kbps/1000;
  preset.setAttribute('aria-label',`${o.name} preset`);mbit.setAttribute('aria-label',`${o.name} bitrate, Mbit/s`);
  preset.oninput=()=>{encode.preset=preset.value;onChange();};mbit.oninput=onChange;
  if(count) {
   const input=document.createElement('input');input.id=count.id;input.type='number';input.min='0';input.step='1';
   input.setAttribute('aria-label',count.label);input.oninput=()=>count.onChange(input.value);
   row.querySelector('.count').replaceWith(input);
  }
  return row;
 }
};
