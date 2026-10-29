import App from './App.svelte';

// Golden Layout's structural styles; its look (the mixer palette) is in styles.css.
import 'golden-layout/dist/css/goldenlayout-base.css';
import './styles.css';

const app = new App({
  target: document.getElementById('app')
});

export default app;
