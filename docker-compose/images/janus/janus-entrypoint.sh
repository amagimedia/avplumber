#!/usr/bin/env bash
set -euo pipefail

: "${JANUS_HOST_IP:?set JANUS_HOST_IP to the externally reachable host IP}"

readonly config_dir=/opt/janus-avp/etc/janus
readonly http_port="${JANUS_HTTP_PORT:-8088}"
readonly audio_enabled="${JANUS_AUDIO_ENABLED:-true}"
readonly audio_port="${JANUS_AUDIO_PORT:-5002}"
readonly audio_rtcp_port="${JANUS_AUDIO_RTCP_PORT:-5003}"
readonly video_port="${JANUS_VIDEO_PORT:-5004}"
readonly video_rtcp_port="${JANUS_VIDEO_RTCP_PORT:-5005}"
readonly rtp_port_range="${JANUS_RTP_PORT_RANGE:-20000-20100}"
readonly debug_level="${JANUS_DEBUG_LEVEL:-4}"

[[ -f "${config_dir}/janus.transport.http.jcfg.template" ]] || \
    cp "${config_dir}/janus.transport.http.jcfg" "${config_dir}/janus.transport.http.jcfg.template"
[[ -f "${config_dir}/janus.plugin.streaming.jcfg.template" ]] || \
    cp "${config_dir}/janus.plugin.streaming.jcfg" "${config_dir}/janus.plugin.streaming.jcfg.template"

sed \
    -e "s/__JANUS_HOST_IP__/${JANUS_HOST_IP}/g" \
    -e "s/__JANUS_HTTP_PORT__/${http_port}/g" \
    -e "s/__JANUS_AUDIO_PORT__/${audio_port}/g" \
    -e "s/__JANUS_AUDIO_RTCP_PORT__/${audio_rtcp_port}/g" \
    -e "s/__JANUS_VIDEO_PORT__/${video_port}/g" \
    -e "s/__JANUS_VIDEO_RTCP_PORT__/${video_rtcp_port}/g" \
    -e "s/__JANUS_RTP_PORT_RANGE__/${rtp_port_range}/g" \
    -e "s/__JANUS_DEBUG_LEVEL__/${debug_level}/g" \
    "${config_dir}/janus.jcfg.template" > "${config_dir}/janus.jcfg"

sed \
    -e "s/__JANUS_HTTP_PORT__/${http_port}/g" \
    "${config_dir}/janus.transport.http.jcfg.template" > "${config_dir}/janus.transport.http.jcfg.rendered"
mv "${config_dir}/janus.transport.http.jcfg.rendered" "${config_dir}/janus.transport.http.jcfg"

sed \
    -e "s/__JANUS_AUDIO_ENABLED__/${audio_enabled}/g" \
    -e "s/__JANUS_AUDIO_PORT__/${audio_port}/g" \
    -e "s/__JANUS_AUDIO_RTCP_PORT__/${audio_rtcp_port}/g" \
    -e "s/__JANUS_VIDEO_PORT__/${video_port}/g" \
    -e "s/__JANUS_VIDEO_RTCP_PORT__/${video_rtcp_port}/g" \
    "${config_dir}/janus.plugin.streaming.jcfg.template" > "${config_dir}/janus.plugin.streaming.jcfg.rendered"
mv "${config_dir}/janus.plugin.streaming.jcfg.rendered" "${config_dir}/janus.plugin.streaming.jcfg"

exec /opt/janus-avp/bin/janus \
    -F "${config_dir}" \
    -C "${config_dir}/janus.jcfg" \
    -i "${JANUS_HOST_IP}" \
    -r "${rtp_port_range}" \
    -d "${debug_level}" \
    -o \
    "$@"
