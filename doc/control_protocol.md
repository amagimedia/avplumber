## Control protocol
Control protocol loosely follows [MVCP](https://www.mltframework.org/docs/mvcp/).

On new connection, server (avplumber) sends a line: `100 VTR READY`

Client issues a command followed by arguments separated by spaces and ending with the new line character. The last argument may contain spaces.

Server (avplumber) responds with line with status code and information:
* `200 OK` - command accepted, empty response
* `201 OK` - command accepted, response will follow and an empty line marks the end of response
* `400 Unknown command: ...`
* `500 ERROR: ...`
* `BYE` and connection close - special response for `bye` command


## Commands

```hello```

Replies with `HELLO`

```version```

Replies with app version and build date

```bye```

Closes the connection

### Nodes management & control

```node.add { ...json object... }```

Add node

```node.add_create { ...json object... }```

Add and create node (without starting it right now)

```node.add_start { ...json object... }```

Add, create and start node

```node.delete name```

Delete node

```node.start name```

Start node

```node.stop name```

Stop node (`auto_restart` action is inhibited)

```node.stop_wait name```

Stop node, return reply when node really stopped

```node.auto_restart name```

Stop node and trigger its `auto_restart` action. This command is probably most useful for restarting input after changing its URL, with confidence that it will be handled the same way as if the previous input stream finished. And unlike `retry group.restart ...`, nothing blocks! See also more experimental `node.interrupt`.

```node.interrupt name```

Stop node *even if it is being constructed right now*. Also, bypass any locks. Currently only `input` node supports this command. After interruption, `auto_restart` action will be triggered.

```node.param.set node_name param_name new_json_value```

Change node parameter. Equivalent in JavaScript: `node['param_name'] = JSON.parse('new_json_value')`. **WARNING:** Node won't accept new parameters until restarted.

```node.param.get node_name```

Get whole node JSON object, for example:
```
node.param.get encode720
{"codec":"libx264","dst":"venc1","group":"out","name":"encode720","options":{"b":"4M","bufsize":"8M","flags":"+cgop","g":25,"level":"3.2","maxrate":"5M","minrate":"3M","muxdelay":0,"muxrate":0,"preset":"ultrafast","profile":"baseline","sc_threshold":0,"x264opts":"no-scenecut"},"src":"vs1o","type":"enc_video"}
OK
```

```node.param.get node_name param_name```

Get single parameter as JSON. Example:

```
node.param.get encode720 group
"out"
OK
```

```node.object.get node_name object_name```

Get object from node. Example:
```
node.object.get input streams
201 OK
[{"codec":"h264","index":0,"type":"V"},{"codec":"h264","index":1,"type":"V"},{"codec":"h264","index":2,"type":"V"},{"codec":"aac","index":3,"type":"A"},{"codec":"aac","index":4,"type":"A"},{"codec":"aac","index":5,"type":"A"}]

node.object.get input programs
201 OK
[{"index":0,"streams":[0,3,4,5]},{"index":1,"streams":[1,3,4,5]},{"index":2,"streams":[2,3,4,5]}]
```

```node.object.set node_name object_name object_value```

Set object in node. Example:
```
node.object.set input stream-limits {"start": "5:20", "stop": "5:30", "loop": False }
200 OK
```

### Closed Captions control

```cc.pause```

Pause Closed Captions processing. This command will return an error when no `extract_cc_data` node present or when there are multiple `extract_cc_data` nodes connected in graph.

```cc.resume```

Resume Closed Captions processing. This command will return an error when no `extract_cc_data` node present or when there are multiple `extract_cc_data` nodes connected in graph.

### Queues (edges)

```queue.plan_capacity queue_name capacity```

Plan capacity of queue (which must be created **after** issuing this command). `capacity` is positive integer. **Warning**: [moodycamel::ReaderWriterQueue](https://github.com/cameron314/readerwriterqueue) (the library we use for queues) forces the queue size to be the smallest 2^n-1 (n=integer) larger or equal to specified capacity.

Default capacity can also be changed - use `*` as a queue_name, e.g. `queue.plan_capacity * 7`


```queues.stats```

Print (human-readable) statistics of queues (graph edges). Example:
```
[#...................]  1821.1  videoin
[....................]  243265  aenc1
[....................]  243265  aenc0
[....................]  243265  aenc2
[....................]  243265  mux2
[#####...............]  243265  aenc
[....................]  1821.03  audioin
 ^                      ^         ^
 queue fill             last PTS  queue name
```

```queue.drain queue_name```

Wait until queue is empty.

### Groups

```group.restart group```

```group.stop group```

```group.start group```

### Raw outputs

```output.start output_group```

```output.stop output_group```

### realtime nodes

```realtime.team.reset team```

Manually trigger reset of a realtime team.

### Playback control

```pause team_name now```

Tell all `pause` nodes in a team `team_name` to pause (stop passing packets)

```pause team_name at timestamp```

Tell all `pause` nodes in a team `team_name` to pause (stop passing packets) at specified timestamp.
Timestamp may be expressed in many forms like: `01:02:03` (hh:mm:ss), `01:00.150` (mm:ss.millis), `12000` (time expressed in ms), `2024-10-03T08:12:44.100` (wallclock time, ISO 9601 format with optional milliseconds).

```resume team_name```

Tell all `pause` nodes in a team `team_name` to resume playback

```seek team_name live```

Flush all queues between the `input_res` nodes and the nodes in a `team_name` team and seek to live video.
Live wideo is a video playback from recording which is still recorded. It is delayed 10 seconds after current time.

```seek team_name now timestamp```

Flush all queues between the `input_rec` nodes and the nodes in a `team_name` team and seek to given timestamp.
Timestamp may be expressed in many forms like: `01:02:03` (hh:mm:ss), `01:00.150` (mm:ss.millis), `12000` (time expressed in ms), `2024-10-03T08:12:44.100` (wallclock time, ISO 9601 format with optional milliseconds).
When timestamp is preceeded with `+` sign, then it will seek to `<current time> + <timestamp>` (like `seek team +0:10` - seek 10s forward from now).
When timestamp is preceeded with `-` sign, then it will seek to `<current time> + <timestamp>` (like `seek team frame -0:10` - seek 10s backward from now).
When there is no sign before the timestamp, it will seek go absolute timestamp.

```seek team_name frame <frame number>```

Flush all queues between the `input_rec` nodes and the nodes in a `team_name` team and seek to given frame number.
When frame number is preceeded with `+` sign, then it will seek to `<current frame> + <frame number>` (like `seek team frame +10` - seek 10 frames forward from now).
When frame number is preceeded with `-` sign, then it will seek to `<current frame> + <frame number>` (like `seek team frame -10` - seek 10 frames backward from now).
When there is no sign before the frame number, it will seek go absolute frame number (starting from `0`).

```seek team_name at timestamp_when timestamp_to```

Flush all queues between the `input_rec` nodes and the nodes in a `team_name` team and seek to given `timestamp_to` when playback time reaches `timestamp_when`.
Multiple `seek at` commands may be specified and it will executed in order of adding.
Timestamp may be expressed in many forms like: `01:02:03` (hh:mm:ss), `01:00.150` (mm:ss.millis), `12000` (time expressed in ms), `2024-10-03T08:12:44.100` (wallclock time, ISO 9601 format with optional milliseconds).

```seek team_name clear```

Clears `seek at` queue.

```speed.set team_name speed```

Set speed of the `speed` nodes belonging to the team `team_name` to `speed` (float).

```speed.get team_name```

Get current playback speed of the team `team_name`.

### Teams synchronization

```team.link<Pause> pause-team-A pause-team-B```
```team.link<Speed> speed-team-A speed-team-B```
```team.link<RealTime> realtime-team-A realtime-team-B```
```team.unlink<Pause> pause-team-A pause-team-B```
```team.unlink<Speed> speed-team-A speed-team-B```
```team.unlink<RealTime> realtime-team-A realtime-team-B```

It is possible to dynamically link & unlink teams to act as a single "super-team".
When teams are linked, commands like ```speed.set``` are automatically sent to all linked teams.
Linking teams will also copy team's current status (like pause status) to linked team (from ```team A``` to ```team B```).

### Hardware acceleration

```hwaccel.init { "name": "name", "type": "type" }```

Init hardware accelerator which may be used for encoding, decoding or filtering video frames.
* name - identifier, supports global objects syntax (`@`). If accelerator with given name already exists, it isn't touched and no error is returned.
* type - currently only `cuda` is supported

### Statistics

```stats.subscribe { ... json object ... }```

Subscribe to statistics. JSON object structure:
```
{
  "url":"",         // URL to put statistics using HTTP POST. Empty to write them to console.
  "name":"qwerty",  // opaque
  "interval":1,     // refresh interval, in seconds
  "streams":{
    "audio":[
      {
        "q_pre_dec":"audioin",    // name of queue before decoder (kbitrate, speed, AV_diff are taken from it)
        "q_post_dec":"a10",       // name of queue after decoder (fps, width, height, pix_fmt, field_order, samplerate, samplerate_md, channel_layout are taken from it)
        "decoder":"Audio_Decode"  // name of decoder node (codec, type are taken from it)
      }
    ],
    "video":[
      {
        "q_pre_dec":"videoin",
        "q_post_dec":"v05",
        "decoder":"Video_Decode"
      }
    ]
  },
  "sentinel":"Video_Sentinel"  // name of sentinel node (card flag is taken from it)
}
```

### Events

```event.on.node.finished event_name node_name```

When node finishes, signal (wake up) an event `event_name`

```event.wait event_name```

Wait for event `event_name`

### Special commands

<code>retry <i>command arguments ...</i></code>

Repeat command until it succeeds. Intended for startup scripts (`--script`). Not recommended for remote control.

<code>detach <i>command arguments ...</i></code>

Return 200 OK immediately, run command in background thread.

