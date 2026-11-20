#!/usr/bin/env bash
# Install the official Linux SDK for an x86-64 host or FFmpeg build container.
set -euo pipefail
[[ $(uname -sm) == 'Linux x86_64' ]] || { echo "NDI setup requires x86-64 Linux" >&2; exit 1; }
dest=/opt/ndi/6.4
if [[ -f "$dest/include/Processing.NDI.Lib.h" && -f "$dest/lib/libndi.so.6.4.0" && -e "$dest/lib/libndi.so" ]]; then
    exit 0
fi
mkdir -p /opt/ndi
work=$(mktemp -d /opt/ndi/.install-XXXXXX)
trap 'rm -rf "$work"' EXIT
cd "$work"
curl -fsSL --retry 3 https://downloads.ndi.tv/SDK/NDI_SDK_Linux/Install_NDI_SDK_v6_Linux.tar.gz -o sdk.tgz
echo '1b066cca076d3e8b49a5f1dc7fbf77065967256807e3cb5ea387c574176eace6  sdk.tgz' | sha256sum -c -
tar -xzf sdk.tgz
PAGER=cat bash Install_NDI_SDK_v6_Linux.sh <<< y >/dev/null
mkdir -p sdk/lib
cp -a "NDI SDK for Linux/include" sdk/
cp -a "NDI SDK for Linux/lib/x86_64-linux-gnu"/libndi.so* sdk/lib/
test -e sdk/lib/libndi.so.6.4.0
mv -T sdk "$dest"
