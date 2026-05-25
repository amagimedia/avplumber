#!/usr/bin/env bash
set -euo pipefail

: "${JANUS_HOST_IP:?set JANUS_HOST_IP to the externally reachable host IP}"

readonly config_dir=/opt/janus-avp/etc/janus
readonly rtp_port_range="${JANUS_RTP_PORT_RANGE:-20000-20100}"
readonly debug_level="${JANUS_DEBUG_LEVEL:-4}"

sed \
    -e "s/__JANUS_HOST_IP__/${JANUS_HOST_IP}/g" \
    -e "s/__JANUS_RTP_PORT_RANGE__/${rtp_port_range}/g" \
    -e "s/__JANUS_DEBUG_LEVEL__/${debug_level}/g" \
    "${config_dir}/janus.jcfg.template" > "${config_dir}/janus.jcfg"

exec /opt/janus-avp/bin/janus \
    -F "${config_dir}" \
    -C "${config_dir}/janus.jcfg" \
    -i "${JANUS_HOST_IP}" \
    -r "${rtp_port_range}" \
    -d "${debug_level}" \
    -o \
    "$@"
