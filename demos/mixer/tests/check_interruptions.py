"""Exercise all transition interruption pairs against a running mixer demo."""
import argparse
import asyncio
import json
import pathlib
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from avpmixer.control import AvpConnection
from avpmixer.control import mixer_command

async def run(args):
    c = AvpConnection(args.host, args.port)
    await asyncio.wait_for(c.connect(), 5)
    results = []
    async def status():
        return json.loads(await c.command(f'mixer.status {args.mixer}'))
    async def settle(scene, timeout=5):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            s=await status()
            if s['transition']=='idle' and s['pgm_scene']==scene: return s
            await asyncio.sleep(.02)
        raise AssertionError(('did not settle', scene, s))
    async def take(kind, scene, delay=None):
        payload={'scene':scene}
        if kind=='fade':payload['duration_sec']=1.2
        if kind=='wipe':payload['wipe_file']=args.wipe_file
        if delay is not None:
            payload['start_pts_ms']=(await status())['now_pts_ms']+delay
        await c.command(mixer_command(kind,args.mixer,**payload))
    try:
        for gap in [.05,.45,.95]:
            for first in ['cut','fade','wipe']:
                for second in ['cut','fade','wipe']:
                    await take('cut','fullscreen_0')
                    await settle('fullscreen_0')
                    await asyncio.sleep(.15)
                    await take(first,'grid_4_page_0',1500 if first=='cut' else None)
                    await asyncio.sleep(gap)
                    before=await status()
                    started=time.monotonic()
                    await take(second,'grid_2_page_0')
                    ack=(time.monotonic()-started)*1000
                    await settle('grid_2_page_0')
                    settled=(time.monotonic()-started)*1000
                    # Let the original request's deadline/cleanup pass.
                    await asyncio.sleep(max(.2,1.7-gap-settled/1000))
                    final=await status()
                    assert final['pgm_scene']=='grid_2_page_0' and final['transition']=='idle', final
                    result={'first':first,'second':second,'gap_ms':gap*1000,
                            'interrupted_mode':before['transition'],'ack_ms':round(ack,2),
                            'settled_ms':round(settled,2),'epoch':time.time()}
                    results.append(result)
                    print(json.dumps(result),flush=True)
                    if args.results:
                        args.results.write_text(json.dumps(results,indent=2) + "\n")
    finally: await c.disconnect()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7777)
    parser.add_argument('--mixer', default='mixer')
    parser.add_argument('--wipe-file', required=True, help='Transparent media path on the mixer host')
    parser.add_argument('--results', type=pathlib.Path)
    asyncio.run(run(parser.parse_args()))
