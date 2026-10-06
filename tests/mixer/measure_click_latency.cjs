// Measures warm CUT transitions through the real browser TUI and Janus player.
// See ../docs/latency.md for setup, calibration and interpretation.
const fs=require('fs');
const [endpoint='http://127.0.0.1:19222',videoUrl='http://127.0.0.1:18080/',tuiUrl='http://127.0.0.1:17681/',output='click-latency.json',fullscreenPoint='95,385',gridPoint='395,385',cutPoint='95,575']=process.argv.slice(2);
const directPoint=process.env.DIRECT_POINT;
const point=value=>value.split(',').map(Number);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function connect(page){
 const address=new URL(page.webSocketDebuggerUrl);address.host=new URL(endpoint).host;
 const ws=new WebSocket(address);await new Promise((r,j)=>{ws.addEventListener('open',r,{once:true});ws.addEventListener('error',j,{once:true})});
 let id=0;const pending=new Map();
 ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(!p)return;pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}});
 const call=(method,params={})=>new Promise((resolve,reject)=>{
  const request=++id;
  const timer=setTimeout(()=>{pending.delete(request);reject(Error('Browser timeout: '+method));},10000);
  pending.set(request,{resolve:v=>{clearTimeout(timer);resolve(v)},reject:e=>{clearTimeout(timer);reject(e)}});
  ws.send(JSON.stringify({id:request,method,params}));
 });
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 return {ws,call,evaluate};
}
function observe(){
 const video=document.querySelector('video');if(!video?.requestVideoFrameCallback)throw Error('Video callback API unavailable');
 const canvas=document.createElement('canvas');canvas.width=32;canvas.height=4;
 const ctx=canvas.getContext('2d',{willReadFrequently:true});ctx.imageSmoothingEnabled=false;
 const frames=[];window.__clickProbe={frames,stop:false};
 function crop(capacity,row){
  const w=1080,h=1920/capacity;
  const fw=Math.floor(Math.min(w,h*16/9)/2)*2,fh=Math.floor(fw*9/16/2)*2;
  const x=Math.floor((w-fw)/2),y=Math.floor((h-fh)/2);
  ctx.drawImage(video,x+64*fw/640,y+140*fh/360,512*fw/640,40*fh/360,0,row*2,32,2);
 }
 function code(pixels,capacity,row){
  let value=0;
  for(let bit=0;bit<32;bit++){
   const a=(row*2*32+bit)*4,b=((row*2+1)*32+bit)*4;
   const top=(pixels[a]+pixels[a+1]+pixels[a+2])/3, bottom=(pixels[b]+pixels[b+1]+pixels[b+2])/3;
   if(Math.abs(top-bottom)<120)return null;
   value=((value<<1)|(top>bottom?1:0))>>>0;
  }
  if((value>>>28)!==10)return null;
  const source=(value>>>24)&15,frame=(value>>>8)&65535;
  if(((source*37+(frame>>>8)+(frame&255))&255)!==(value&255))return null;
  return {capacity,source,frame};
 }
 function sample(now,m){
  const started=performance.now();
  crop(1,0);crop(4,1);
  const pixels=ctx.getImageData(0,0,32,4).data;
  frames.push({display:performance.timeOrigin+m.expectedDisplayTime,callback:performance.timeOrigin+now,mediaTime:m.mediaTime,presented:m.presentedFrames,codes:[code(pixels,1,0),code(pixels,4,1)].filter(Boolean),readMs:performance.now()-started});
  if(frames.length>10000)frames.shift();
  if(!window.__clickProbe.stop)video.requestVideoFrameCallback(sample);
 }
 video.requestVideoFrameCallback(sample);return true;
}
async function click(page,x,y){
 await page.evaluate(`window.__clickAt=null;document.addEventListener('mousedown',()=>{window.__clickAt=performance.timeOrigin+performance.now()},{once:true,capture:true})`);
 await page.call('Input.dispatchMouseEvent',{type:'mousePressed',x,y,button:'left',clickCount:1});
 await page.call('Input.dispatchMouseEvent',{type:'mouseReleased',x,y,button:'left',clickCount:1});
 return page.evaluate('window.__clickAt');
}
(async()=>{
 const pages=await fetch(endpoint+'/json/list').then(r=>r.json());
 const vp=pages.find(p=>p.url===videoUrl);if(!vp)throw Error('Open the video page in the debugging browser');
 const video=await connect(vp);
 await video.call('Emulation.setDeviceMetricsOverride',{width:2200,height:1000,deviceScaleFactor:1,mobile:false});
 await video.call('Page.enable');
 await video.evaluate(`(()=>{document.querySelector('main').style.width='600px';let f=document.getElementById('latencyTui');if(!f){f=document.createElement('iframe');f.id='latencyTui';f.style='position:fixed;left:600px;top:0;width:1600px;height:1000px;border:0';document.body.append(f);}f.src=${JSON.stringify(tuiUrl)};})()`);
 await video.call('Page.bringToFront');
 await sleep(1500);
 const tree=await video.call('Page.getFrameTree');const frame=tree.frameTree.childFrames.find(f=>f.frame.url===tuiUrl).frame;
 const world=await video.call('Page.createIsolatedWorld',{frameId:frame.id,worldName:'latency-click'});
 const tui={ws:{close(){}},call:(method,params)=>video.call(method,method==='Input.dispatchMouseEvent'?{...params,x:params.x+600}:params),evaluate:async expression=>{const r=await video.call('Runtime.evaluate',{expression,contextId:world.executionContextId,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}};const result={method:'browser mousedown to expectedDisplayTime of first verified destination frame',mode:directPoint?'direct cut':'preloaded cut',samples:[]};
 try{
  const screenshot=await tui.call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(output+'.tui.png',Buffer.from(screenshot.data,'base64'));
  await click(tui,...point(fullscreenPoint));
  await sleep(200);
  await click(tui,...point(cutPoint));
  await sleep(350);
  await video.evaluate(`(${observe.toString()})()`);
  await sleep(1000);
  const warm=await video.evaluate('window.__clickProbe.frames');
  if(warm.length<20||!warm.some(f=>f.codes.length))throw Error(`Unhealthy video: ${warm.length} callbacks in one second; no measurement accepted`);
  if(directPoint)await click(tui,...point(directPoint));
  // Dedicated TUI tab at 1600x1000, 15px ttyd font. Direct mode must be OFF.
  // Preview each layout before timing the CUT button, so this measures a warm cut.
  for(let trial=0;trial<12;trial++){
   const target=trial%2?4:1;
   if(!directPoint){
    await click(tui,...point(target===1?fullscreenPoint:gridPoint));
    await sleep(350);
   }
   const previous=await video.evaluate('window.__clickProbe.frames.at(-1)');
   if(previous.codes.some(c=>c.capacity===target)){if(trial===0)continue;throw Error('Target already visible before timed click');}
   const at=await click(tui,...point(directPoint?(target===1?fullscreenPoint:gridPoint):cutPoint));if(!at)throw Error('No browser mousedown timestamp');
   let match=null;
   for(let i=0;i<100;i++){
    await sleep(25);
    match=await video.evaluate(`window.__clickProbe.frames.find(f=>f.display>=${at}&&f.codes.some(c=>c.capacity===${target}&&c.source===0))||null`);
    if(match)break;
   }
   if(!match){result.samples.push({trial,target,timeout:true});throw Error('No verified destination frame within 2.5 seconds');}
   const sample={trial,target,latencyMs:match.display-at,callbackLateMs:match.callback-match.display,readMs:match.readMs};
   result.samples.push(sample);console.log(JSON.stringify(sample));await sleep(200);
  }
  const latencies=result.samples.map(s=>s.latencyMs).sort((a,b)=>a-b);
  result.summary={count:latencies.length,minMs:latencies[0],meanMs:latencies.reduce((a,b)=>a+b,0)/latencies.length,medianMs:latencies[Math.floor(latencies.length/2)],p95Ms:latencies[Math.ceil(latencies.length*.95)-1],maxMs:latencies.at(-1)};
  console.log(JSON.stringify(result.summary));
 }catch(e){result.error=String(e);throw e;}
 finally{fs.writeFileSync(output,JSON.stringify(result,null,2));await video.evaluate('if(window.__clickProbe)window.__clickProbe.stop=true');video.ws.close();tui.ws.close();}
})().catch(e=>{console.error(e);process.exit(1)});
