"""Record the mixer demo's Program and Program-preview outputs on the instance, with the cut sequence.

Run on the host that runs the mixer stack: `python3 capture_on_instance.py > capture.json`.
Janus's streaming plugin records the two existing RTP outputs (mountpoints 1 and 5008) as received,
before any Internet transport, into .mjr files inside the Janus container. The script turns all
keys on, cuts through the demo scenes over ten seconds, polls the mixer state, stops the
recordings and restores the scenes and keys it found. It prints one JSON report: the recording
paths, every command with its time, the state snapshots and the deadline and drop counters
before and after. Standard library only; it changes no setup and restarts nothing.
"""
import json,time,uuid,urllib.request,threading
base='http://127.0.0.1:17681'
janus='http://127.0.0.1:8088/janus'
prefix='/tmp/avplumber-native-'+str(int(time.time()))
report={'files':{},'states':[],'events':[]}
def request(url,body=None):
 payload=None if body is None else json.dumps(body).encode()
 with urllib.request.urlopen(urllib.request.Request(url,payload,{'Content-Type':'application/json'}),timeout=8) as r:return json.load(r)
def call(url,body):
 data=request(url,{**body,'transaction':uuid.uuid4().hex})
 if data.get('janus')!='success':raise RuntimeError(data)
 return data
def command(body):
 if body['command'] not in ['dsk','cut','preview']:raise ValueError(body)
 started=time.time(); result=request(base+'/api/command',body)
 if not result.get('ok'):raise RuntimeError(result)
 report['events'].append({'wall':started,'done':time.time(),'command':body})
def snapshot():
 t=time.time();s=request(base+'/api/state');report['states'].append({'wall':t,'received':time.time(),'state':s});return s
original=snapshot();report['original']=original
session=janus+'/'+str(call(janus,{'janus':'create'})['data']['id'])
handle=session+'/'+str(call(session,{'janus':'attach','plugin':'janus.plugin.streaming'})['data']['id'])
def streaming(body):
 data=call(handle,{'janus':'message','body':body})['plugindata']['data']
 if 'error' in data:raise RuntimeError(data)
 return data
started=[]
try:
 for mid,name in [(1,'pgm'),(5008,'preview')]:
  info=streaming({'request':'info','id':mid})['info']
  report[name+'_info']=info
  if info.get('recording') or any(m.get('recording') for m in info['media']):raise RuntimeError('Stream is already being recorded')
  media=next(m for m in info['media'] if m['type']=='video')
  path=prefix+'-'+name
  streaming({'request':'recording','action':'start','id':mid,'media':[{'mid':media['mid'],'filename':path}]})
  started.append((mid,media['mid']));report['files'][name]=path+'.mjr'
 for key in original['dsk']:command({'command':'dsk','key':key['id'],'on':True,'fade_seconds':0.4})
 command({'command':'cut','scene':'grid_16_080'})
 # Padding guarantees the recording includes complete keyframes and settled keys.
 for _ in range(5):snapshot();time.sleep(1)
 report['demo_start_wall']=time.time();start=time.monotonic()
 sequence=[(1.0,'random_180'),(2.4,'grid_64_112'),(3.9,'pip_128'),(5.4,'alpha_overlay_224'),(7.0,'grid_8_064'),(8.5,'random_179')]
 for at,scene in sequence:
  time.sleep(max(0,start+at-time.monotonic()))
  command({'command':'cut','scene':scene});snapshot()
 time.sleep(max(0,start+10-time.monotonic()))
 report['demo_end_wall']=time.time();snapshot();time.sleep(2)
finally:
 errors=[]
 for mid,media in started:
  try:streaming({'request':'recording','action':'stop','id':mid,'media':[{'mid':media}]})
  except Exception as e:errors.append(str(e))
 for body in [{'command':'cut','scene':original['status']['pgm_scene']},{'command':'preview','scene':original['status']['pvw_scene']},*[{'command':'dsk','key':k['id'],'on':k['on'],'fade_seconds':0} for k in original['dsk']]]:
  try:command(body)
  except Exception as e:errors.append(str(e))
 report['restored']=snapshot()
 report['counters_after']={'playout':report['restored']['status']['playout'],'aux_missed':sum(b['playout']['missed_deadlines'] for b in report['restored']['aux_buses']),'aux_drops':sum(b['output_drops'] for b in report['restored']['aux_buses'])}
 report['counters_before']={'playout':original['status']['playout'],'aux_missed':sum(b['playout']['missed_deadlines'] for b in original['aux_buses']),'aux_drops':sum(b['output_drops'] for b in original['aux_buses'])}
 call(session,{'janus':'destroy'})
 report['cleanup_errors']=errors
 print(json.dumps(report),flush=True)
 if errors:raise RuntimeError(errors)
