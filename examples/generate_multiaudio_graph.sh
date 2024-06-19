#!/bin/bash
AUDIO_STREAMS=$1
IN_URL="$2"
OUT_URL="$3"

for i in `seq 0 1 $[AUDIO_STREAMS-1]`; do
    echo "queue.plan_capacity in_a$i 250"
    echo "queue.plan_capacity Audio_Encode_out$i 250"
done

echo 'queue.plan_capacity in_v_dec 240
node.add {"url":"https://dqktq9708cqse.cloudfront.net/cards/broken_stream_board_v03.jpg","type":"input","auto_restart":"off","name":"slate_input","group":"slate_loader","dst":"slate_input"}
node.add {"routing":{"v:0":"slate_vpkt"},"type":"demux","auto_restart":"off","name":"slate_demux","group":"slate_loader","src":"slate_input"}
node.add {"type":"dec_video","auto_restart":"off","name":"slate_decoder","group":"slate_loader","src":"slate_vpkt","dst":"slate_vfrm"}
node.add {"dst_width":1920,"dst_height":1080,"dst_pixel_format":"yuv420p","flags":["SWS_LANCZOS"],"type":"rescale_video","auto_restart":"off","name":"slate_scaler","group":"slate_loader","src":"slate_vfrm","dst":"slate_final"}
node.add {"buffer":"slate","type":"picture_buffer_sink","auto_restart":"off","name":"slate_buffer_sink","group":"slate_loader","src":"slate_final"}
event.on.node.finished slate_ready slate_buffer_sink
group.start slate_loader
event.wait slate_ready
group.stop slate_loader

node.add {"url":"'"$IN_URL"'","initial_timeout":20,"timeout":10.0,"options":{"probesize":5000000,"analyzeduration":5000000,"fflags":"discardcorrupt"},"type":"input","auto_restart":"group","name":"input","group":"in","dst":"in_mux0"}
'

routing='"v:0":"in_v_prere"'"$(for i in `seq 0 1 $[AUDIO_STREAMS-1]`; do
    printf ',"?a:'$i'":"in_a_prere'$i'"'
done)"

echo 'node.add {"wait_for_keyframe":true,"routing":{'"$routing"'},"type":"demux","auto_restart":"group","name":"demux","group":"in","src":"in_mux0"}'

echo 'node.add {"speed":1.0,"type":"realtime","auto_restart":"group","group":"in","src":"in_v_prere","dst":"in_v"}
node.add {"type":"split","auto_restart":"group","group":"in","src":"in_v","dst":["in_v_dec"]}
node.add {"type":"dec_video","auto_restart":"group","name":"Video_Decode","group":"in","src":"in_v_dec","dst":"Video_Decode_out"}
node.add {"graph":"[v:0]setsar=1/1","type":"filter_video","auto_restart":"group","name":"Video_Filter","group":"in","src":"Video_Decode_out","dst":"Video_Filter_out"}
node.add {"fps":"60000/1001","type":"force_fps","auto_restart":"panic","group":"out","src":"Video_Filter_out","dst":"v_fps_forced"}
node.add {"pixel_format":"yuv420p","type":"assume_video_format","auto_restart":"panic","group":"out","src":"v_fps_forced","dst":"v0"}
node.add {"timeout":1,"freeze":0,"backup_picture_buffer":"slate","reporting_url":"","type":"sentinel_video","auto_restart":"panic","name":"Video_Sentinel","group":"out","src":"v0","dst":"v1"}
node.add {"type":"split","auto_restart":"panic","group":"out","src":"v1","dst":["vs1i"]}
node.add {"dst_width":1920,"dst_height":1080,"flags":["SWS_LANCZOS"],"dst_pixel_format":"yuv420p","type":"rescale_video","auto_restart":"panic","name":"Scale_fhd","group":"out","src":"vs1i","dst":"Scale_fhd_out"}
node.add {"type":"split","auto_restart":"panic","group":"out","src":"Scale_fhd_out","dst":["splittosplit_fhd","splittokeyforce_fhd"]}
node.add {"interval_sec":1,"type":"force_keyframe","auto_restart":"panic","group":"out","src":"splittokeyforce_fhd","dst":"keyframed_fhd"}
node.add {"codec":"libx264","options":{"preset":"ultrafast","b":"8500k","minrate":"7000k","maxrate":"10000k","flags":"+cgop","profile":"main","level":"40","bufsize":"20000k"},"type":"enc_video","auto_restart":"panic","name":"VEncode_fhd","group":"out","src":"keyframed_fhd","dst":"venc_fhd"}  
node.add {"type":"split","auto_restart":"panic","name":"venc_fhd_split","group":"out","src":"venc_fhd","dst":["venc_fhd_hls"]}
'

for i in `seq 0 1 $[AUDIO_STREAMS-1]`; do
    echo '
    node.add {"speed":1.0,"type":"realtime","auto_restart":"group","group":"in","src":"in_a_prere'$i'","dst":"in_a'$i'"}
    node.add {"optional":true,"type":"dec_audio","auto_restart":"group","name":"Audio_Decode'$i'","group":"in","src":"in_a'$i'","dst":"adec'$i'"}
    node.add {"type":"assume_audio_format","auto_restart":"panic","group":"out","src":"adec'$i'","dst":"a0_'$i'"}
    node.add {"timeout":1,"type":"sentinel_audio","auto_restart":"panic","name":"Audio_Sentinel'$i'","group":"out","src":"a0_'$i'","dst":"a1_'$i'"}
    node.add {"dst_sample_rate":48000,"dst_channel_layout":"stereo","dst_sample_format":"fltp","compensation":0,"type":"resample_audio","auto_restart":"panic","name":"Audio_Resample'$i'","group":"out","src":"a1_'$i'","dst":"Audio_Resample_out'$i'"}
    node.add {"type":"split","auto_restart":"panic","group":"out","src":"Audio_Resample_out'$i'","dst":["a0toint'$i'","atoenc'$i'"]}
    node.add {"dst_sample_rate":48000,"dst_channel_layout":"stereo","dst_sample_format":"s16","compensation":0,"type":"resample_audio","auto_restart":"panic","name":"Audio0_to_s16'$i'","group":"out","src":"a0toint'$i'","dst":"Audio0_to_s16_out'$i'"}
    node.add {"codec":"aac","options":{"b":"320k"},"type":"enc_audio","auto_restart":"panic","name":"Audio_Encode'$i'","group":"out","src":"atoenc'$i'","dst":"Audio_Encode_out'$i'"}
    node.add {"type":"split","auto_restart":"panic","group":"out","src":"Audio_Encode_out'$i'","dst":["aenc'$i'"]}
    node.add {"drop":true,"type":"split","auto_restart":"off","name":"RawOutSplit'$i'","group":"out","src":"Audio0_to_s16_out'$i'","dst":[]}
    '
done

aouts="$(for i in `seq 0 1 $[AUDIO_STREAMS-1]`; do
    printf ',"aenc'$i'"'
done)"

echo '
node.add {"type":"mux","auto_restart":"panic","name":"Mux_fhd","group":"out","src":["venc_fhd_hls"'"$aouts"'],"dst":"muxed_v_fhd"}
#node.add {"format":"mpegts","url":"srt://localhost:9000?streamid=publish:output","options":{},"type":"output","auto_restart":"panic","name":"Out_fhd","group":"out","src":"muxed_v_fhd"}
node.add {"format":"mpegts","url":"'"$OUT_URL"'","options":{},"type":"output","auto_restart":"panic","name":"Out_fhd","group":"out","src":"muxed_v_fhd"}

node.add {"drop":true,"type":"split","auto_restart":"off","name":"RawOutSplit_v_fhd","group":"out","src":"splittosplit_fhd","dst":[]}
#stats.subscribe {"url":"","name":"stream1","interval":3,"streams":{"video":[{"decoder":"Video_Decode","q_pre_dec":"in_v_dec","q_post_dec":"Video_Decode_out"}],"audio":[{"decoder":"Audio_Decode","q_pre_dec":"in_a","q_post_dec":"adec"}]},"sentinel":"Video_Sentinel"}   
group.start out
detach retry group.start in
'