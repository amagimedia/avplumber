# OBS avplumber source with zero-copy CUDA rendering

This plugin allows embedding avplumber graph into OBS, as a source. You can use it as a replacement for ffmpeg_source (media source) for live video input (avplumber doesn't support seeking so it isn't that useful for media files).

It requires patched OBS. The patch was created for [OBS 27.0.1](https://github.com/obsproject/obs-studio/tree/27.0.1) and wasn't updated since then. See `obs_patches/hwaccel_async_sources.patch`.

Patched OBS is required only because this plugin supports hardware-accelerated decoding and rendering on NVIDIA hardware, using nvdec/cuvid & CUDA, without frames leaving the GPU and wasting PCIe bandwidth (so called zero-copy). If you don't need this feature, you can strip functionality related to hardware frames in `avplumber/src/nodes/obs/obs_video_sink.cpp`. Start with class member `obs_hw_` and remove code related to it.

## How to build

```
pluginsrc=`pwd`
avpdir=$pluginsrc/../..
obsdir=$HOME/obs27
mkdir $obsdir
git clone --recursive -b 27.0.1 https://github.com/obsproject/obs-studio $obsdir
rsync -a $pluginsrc $obsdir/plugins/
rsync -a --exclude=.git --exclude=library_examples --exclude=objs $avpdir/ $obsdir/plugins/obs-avplumber-source/avplumber
cd $obsdir
patch -p1 < $pluginsrc/obs_patches/add_avplumber_plugin.patch
patch -p1 < $pluginsrc/obs_patches/hwaccel_async_sources.patch
```
and build OBS as usual.

## How to use

Add avplumber source to some scene.

Paste content of `examples/rtmp_input.txt` to the script field in source properties, change input URL and confirm.

After a few seconds, you should see and hear the input stream in OBS.

## Available nodes

### `obs_video_sink`

Outputs video into the video mixer of OBS. Supports software frames and hardware frames (CUDA, VAAPI) with pixel data staying on the GPU, and also `EglImageFrame`.

1 input: `av::VideoFrame` or `EglImageFrame`

This node is non-blocking.

Parameters:
-   `max_freeze_duration` (float, seconds) - output an empty frame after this time without input
-   `unbuffered` (bool, default `false`) - disables frame queue in OBS.

To achieve frame-perfect output:
* specify `tick_source: "obs"` and enable `unbuffered`
* prepend this node with `force_fps` (set to video mixer's FPS) and `realtime` (also specify `tick_source: "obs"` in the latter)

### `obs_audio_sink`

Outputs audio into the audio mixer of OBS.

1 input: `av::AudioSamples`

no parameters
