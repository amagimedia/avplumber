#!/usr/bin/env bash
set -euo pipefail

: "${JANUS_HOST_IP:?set JANUS_HOST_IP to the externally reachable host IP}"

readonly config_dir=/opt/janus-avp/etc/janus
readonly http_port="${JANUS_HTTP_PORT:-8088}"
readonly ws_port="${JANUS_WS_PORT:-8188}"
readonly audio_enabled="${JANUS_AUDIO_ENABLED:-true}"
readonly audio_port="${JANUS_AUDIO_PORT:-5002}"
readonly audio_rtcp_port="${JANUS_AUDIO_RTCP_PORT:-5003}"
readonly video_port="${JANUS_VIDEO_PORT:-5004}"
readonly video_rtcp_port="${JANUS_VIDEO_RTCP_PORT:-5005}"
readonly hdr_video_port="${JANUS_HDR_VIDEO_PORT:-5006}"
readonly hdr_video_rtcp_port="${JANUS_HDR_VIDEO_RTCP_PORT:-5007}"
readonly rtp_port_range="${JANUS_RTP_PORT_RANGE:-20000-20100}"
readonly debug_level="${JANUS_DEBUG_LEVEL:-4}"

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

for transport in http websockets; do
    config="${config_dir}/janus.transport.${transport}.jcfg"
    [[ -f "${config}.template" ]] || cp "${config}" "${config}.template"
    sed -e "s/__JANUS_HTTP_PORT__/${http_port}/g" -e "s/__JANUS_WS_PORT__/${ws_port}/g" \
        "${config}.template" > "${config}.rendered"
    mv "${config}.rendered" "${config}"
done

sed \
    -e "s/__JANUS_AUDIO_ENABLED__/${audio_enabled}/g" \
    -e "s/__JANUS_AUDIO_PORT__/${audio_port}/g" \
    -e "s/__JANUS_AUDIO_RTCP_PORT__/${audio_rtcp_port}/g" \
    -e "s/__JANUS_VIDEO_PORT__/${video_port}/g" \
    -e "s/__JANUS_VIDEO_RTCP_PORT__/${video_rtcp_port}/g" \
    -e "s/__JANUS_HDR_VIDEO_PORT__/${hdr_video_port}/g" \
    -e "s/__JANUS_HDR_VIDEO_RTCP_PORT__/${hdr_video_rtcp_port}/g" \
    "${config_dir}/janus.plugin.streaming.jcfg.template" > "${config_dir}/janus.plugin.streaming.jcfg.rendered"
mv "${config_dir}/janus.plugin.streaming.jcfg.rendered" "${config_dir}/janus.plugin.streaming.jcfg"

# Behind 1:1 NAT (a cloud VM: the public IP is not on any interface), JANUS_HOST_IP is the
# interface address and JANUS_NAT_1_1 the public one Janus puts in its ICE candidates. A stale
# one means no WebRTC media at all, and a cloud VM's ephemeral public IP changes when it stops
# and starts while the container keeps its environment: JANUS_NAT_1_1=auto reads the address
# from the instance metadata (GCE, then AWS) at every start, a restart after a reboot included.
cloud_public_ip() {
    local ip token
    ip=$(curl -fsS -m 2 -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null) ||
        { token=$(curl -fsS -m 2 -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
              http://169.254.169.254/latest/api/token 2>/dev/null) &&
          ip=$(curl -fsS -m 2 -H "X-aws-ec2-metadata-token: ${token}" \
              http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null); } || return 1
    [[ ${ip} =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] && printf '%s' "${ip}"
}

nat_1_1="${JANUS_NAT_1_1:-}"
if [[ ${nat_1_1} == auto ]]; then
    nat_1_1=""
    for _ in 1 2 3 4 5 6 7 8 9 10; do   # the metadata server can lag the network at boot
        nat_1_1=$(cloud_public_ip) && break
        sleep 2
    done
    if [[ -n ${nat_1_1} ]]; then
        echo "janus-entrypoint: public address ${nat_1_1} from the instance metadata"
    else
        echo "janus-entrypoint: JANUS_NAT_1_1=auto found no public address in GCE or AWS instance" \
             "metadata; viewers outside this network get no media" >&2
    fi
fi
[[ ${1:-} != --print-public-address ]] || { printf '%s\n' "${nat_1_1}"; exit 0; }
nat_args=()
[[ -z "${nat_1_1}" ]] || nat_args=(-1 "${nat_1_1}")

exec /opt/janus-avp/bin/janus \
    -F "${config_dir}" \
    -C "${config_dir}/janus.jcfg" \
    -i "${JANUS_HOST_IP}" \
    -r "${rtp_port_range}" \
    -d "${debug_level}" \
    -o \
    "${nat_args[@]}" \
    "$@"
