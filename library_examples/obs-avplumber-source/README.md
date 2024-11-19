# OBS avplumber source with zero-copy CUDA rendering

This plugin allows embedding avplumber graph into OBS, as a source. You can use it as a replacement for ffmpeg_source (media source) for live video input (avplumber doesn't support seeking so it isn't that useful for media files).

It requires patched OBS. The patches were created for [OBS 27.0.1](https://github.com/obsproject/obs-studio/tree/27.0.1) and [OBS 31.0.0-beta1 commit ba6a6bfd](https://github.com/obsproject/obs-studio/tree/ba6a6bfdcb4db6fdee02b0224df4106321c5ef48). See `obs$VERSION_patches/hwaccel_async_sources.patch`.

Patched OBS is required only because this plugin supports hardware-accelerated decoding and rendering using nvdec/cuvid & CUDA (NVIDIA) or VA-API (AMD, possibly other vendors but not tested yet), without frames leaving the GPU and wasting PCIe bandwidth (so called zero-copy). If you don't need this feature, you can strip functionality related to hardware frames in `avplumber/src/nodes/obs/obs_video_sink.cpp`. Start with class member `obs_hw_` and remove code related to it.

## How to build

### OBS 27

```
pluginsrc=`pwd`
avpdir=$pluginsrc/../..
obsdir=$HOME/obs27
cp $pluginsrc/CMakeLists.obs27.txt $pluginsrc/CMakeLists.txt
mkdir $obsdir
git clone --recursive -b 27.0.1 https://github.com/obsproject/obs-studio $obsdir
rsync -a $pluginsrc $obsdir/plugins/
rsync -a --exclude=.git --exclude=library_examples --exclude=objs $avpdir/ $obsdir/plugins/obs-avplumber-source/avplumber
cd $obsdir
patch -p1 < $pluginsrc/obs27_patches/add_avplumber_plugin.patch
patch -p1 < $pluginsrc/obs27_patches/hwaccel_async_sources.patch
```
and build OBS as usual.

### OBS 31

```
pluginsrc=`pwd`
avpdir=$pluginsrc/../..
obsdir=$HOME/obs31
cp $pluginsrc/CMakeLists.obs31.txt $pluginsrc/CMakeLists.txt
mkdir $obsdir
git clone --recursive -b 31.0.0-beta1 https://github.com/obsproject/obs-studio $obsdir
rsync -a $pluginsrc $obsdir/plugins/
rsync -a --exclude=.git --exclude=library_examples --exclude=objs $avpdir/ $obsdir/plugins/obs-avplumber-source/avplumber
cd $obsdir
patch -p1 < $pluginsrc/obs31_patches/add_avplumber_plugin.patch
patch -p1 < $pluginsrc/obs31_patches/hwaccel_async_sources.patch
```
and build OBS as usual. (TODO: test whether these instructions work)

## How to use

Add avplumber source to some scene.

Paste content of `examples/rtmp_input.txt` to the script field in source properties, change input URL and confirm.

After a few seconds, you should see and hear the input stream in OBS.
