const reel = document.getElementById('reel');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const play = () => reel.play().catch(() => {});
if (!reduceMotion.matches) play();
document.getElementById('replay').addEventListener('click', () => {
  reel.currentTime = 0;
  play();
});
document.querySelectorAll('[data-seek]').forEach(button => {
  button.addEventListener('click', () => {
    const seek = () => { reel.currentTime = Number(button.dataset.seek); play(); };
    if (reel.readyState) seek();
    else { reel.addEventListener('loadedmetadata', seek, {once: true}); reel.load(); }
    document.getElementById('demo').scrollIntoView({behavior: reduceMotion.matches ? 'auto' : 'smooth', block: 'start'});
  });
});
