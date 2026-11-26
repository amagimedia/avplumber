# Sourced by desktop.sh. Capture the operator's display before starting Compose.
desktop_display() {
    local wayland="${WAYLAND_DISPLAY:-}" display="${DISPLAY:-}" display_number
    if [[ -n "$wayland" && "$wayland" != /* && -n "${XDG_RUNTIME_DIR:-}" ]]; then
        wayland="$XDG_RUNTIME_DIR/$wayland"
    fi
    if [[ "$wayland" == /* && -S "$wayland" ]]; then
        export MIXER_GUI_PLATFORM=wayland MIXER_GUI_DISPLAY_SOURCE="$wayland"
        export MIXER_GUI_DISPLAY_TARGET=/tmp/mixer-wayland MIXER_GUI_WAYLAND=/tmp/mixer-wayland
        export MIXER_GUI_XDISPLAY= MIXER_GUI_AUTH=/dev/null
    elif [[ "$display" =~ ^(:|unix:)([0-9]+)(\.[0-9]+)?$ ]]; then
        display_number="${BASH_REMATCH[2]}"
        export MIXER_GUI_PLATFORM=x11 MIXER_GUI_XDISPLAY=":$display_number" MIXER_GUI_WAYLAND=
        export MIXER_GUI_DISPLAY_SOURCE="/tmp/.X11-unix/X$display_number"
        export MIXER_GUI_DISPLAY_TARGET="$MIXER_GUI_DISPLAY_SOURCE"
        export MIXER_GUI_AUTH="${XAUTHORITY:-$HOME/.Xauthority}"
        [[ -S "$MIXER_GUI_DISPLAY_SOURCE" && -r "$MIXER_GUI_AUTH" ]] || return 1
    else
        return 1
    fi
}

local_graphical_login() {
    [[ -z "${SSH_CONNECTION:-}${SSH_TTY:-}" ]] || return 1
    local session="${XDG_SESSION_ID:-}" props
    if command -v loginctl >/dev/null; then
        [[ -n "$session" ]] || session=$(loginctl show-user "$(id -u)" -p Display --value 2>/dev/null) || return 1
        [[ -n "$session" ]] || return 1
        props=$(loginctl show-session "$session" -p User -p Active -p Remote -p Seat -p Type 2>/dev/null) || return 1
        props=$'\n'"$props"$'\n'
        [[ "$props" == *$'\n'"User=$(id -u)"$'\n'* && "$props" == *$'\nActive=yes\n'* &&
           "$props" == *$'\nRemote=no\n'* && "$props" =~ $'\nSeat='[^[:space:]]+ &&
           "$props" =~ $'\nType='(wayland|x11)$'\n' ]]
    else
        [[ -n "${XDG_SEAT:-}" && "${XDG_SESSION_TYPE:-}" =~ ^(wayland|x11)$ ]]
    fi
}

configure_gui() {
    case "$1" in
        off) return 1 ;;
        auto) local_graphical_login && desktop_display || return 1 ;;
        on) desktop_display || fail "launch the GUI from a Wayland or X11 desktop with a readable display socket" ;;
        *) fail "MIXER_DESKTOP_UI must be auto, on or off" ;;
    esac
    [[ $(id -u) != 0 ]] || fail "launch the desktop app as the logged-in desktop user"
    export MIXER_GUI_UID=$(id -u) MIXER_GUI_GID=$(id -g)
    export MIXER_GUI_RENDER_GID=$(stat -c %g "$WLR_RENDER_DRM_DEVICE") MIXER_GUI_HOSTNAME=$(hostname)
    export MIXER_GUI_PROFILE="${XDG_STATE_HOME:-$HOME/.local/state}/avplumber/$COMPOSE_PROJECT_NAME"
    mkdir -p -m 700 "$MIXER_GUI_PROFILE"
}
