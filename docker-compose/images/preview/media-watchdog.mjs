// Signalling and readyState can remain healthy after the last video frame. Watch
// presentation independently of getStats(), which may itself stop resolving.
export function createMediaWatchdog(video, {
  onPlaying, onStall,
  now = () => performance.now(),
  timers = window,
  visibility = document,
  startupMs = 12000,
  stallMs = 8000,
} = {}) {
  let stopped = false, attached = false, playing = false;
  let progressAt = now(), previousTime = video.currentTime;
  let timer = null, frame = null;
  const suspended = () => visibility.hidden || (playing && video.paused && !video.ended);
  const resetGrace = () => { progressAt = now(); previousTime = video.currentTime; };
  const presented = () => {
    if (stopped) return;
    progressAt = now();
    if (!playing) {
      playing = true;
      onPlaying();
    }
  };
  const nextFrame = () => {
    frame = video.requestVideoFrameCallback(() => {
      if (stopped) return;
      presented();
      if (!stopped) nextFrame();
    });
  };
  function stop() {
    stopped = true;
    timers.clearTimeout(timer);
    if (frame !== null) video.cancelVideoFrameCallback?.(frame);
    visibility.removeEventListener("visibilitychange", resetGrace);
    video.removeEventListener("playing", resetGrace);
    video.removeEventListener("pause", resetGrace);
  }
  function check() {
    if (stopped) return;
    if (suspended()) resetGrace();
    else {
      // Older browsers lack presentation callbacks. Require movement of the
      // media clock, never just HAVE_CURRENT_DATA from a retained old frame.
      if (attached && !video.requestVideoFrameCallback) {
        if (video.currentTime > previousTime) presented();
        previousTime = video.currentTime;
      }
      if (now() - progressAt >= (playing ? stallMs : startupMs)) {
        stop();
        onStall(new Error(playing ? "Video stalled for 8 seconds" : "No video received within 12 seconds"));
        return;
      }
    }
    timer = timers.setTimeout(check, 1000);
  }
  visibility.addEventListener("visibilitychange", resetGrace);
  video.addEventListener("playing", resetGrace);
  video.addEventListener("pause", resetGrace);
  timer = timers.setTimeout(check, 1000);
  return {
    get playing() { return playing; },
    attach() {
      if (attached || stopped) return;
      attached = true;
      previousTime = video.currentTime;
      if (video.requestVideoFrameCallback) nextFrame();
    },
    stop,
  };
}
