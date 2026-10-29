const reel = document.querySelector('#reel');
for (const button of document.querySelectorAll('[data-seek]')) {
  button.addEventListener('click', () => {
    reel.currentTime = Number(button.dataset.seek);
    reel.play().catch(() => {});
    reel.scrollIntoView({block: 'center', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  });
}
document.querySelector('#replay').addEventListener('click', () => {
  reel.currentTime = 0;
  reel.play().catch(() => {});
});
const descriptions = {
  overview: 'Overview · 105 NVDEC sources, 40 browser sources and 47 raw feeds.',
  groups: 'Groups · A browser pipeline selected among the show’s processing groups.',
  nodes: 'Nodes · Individual stages and routed connections, with the selected path highlighted.',
  details: 'Details · DMA-BUF import to CUDA, frame rate, queue fill and the node inspector.'
};
for (const button of document.querySelectorAll('[data-view]')) {
  button.addEventListener('click', () => {
    const view = button.dataset.view;
    const url = `https://github.com/amagimedia/avplumber/releases/download/webui-demo-media-2026-10/webgl-192-${view}-20261029.png`;
    for (const other of document.querySelectorAll('[data-view]')) other.setAttribute('aria-pressed', String(other === button));
    const shot = document.querySelector('#shot');
    shot.src = url;
    shot.alt = descriptions[view];
    document.querySelector('#shot-link').href = url;
    document.querySelector('#shot-download').href = url;
    document.querySelector('#shot-caption').textContent = descriptions[view];
  });
}
