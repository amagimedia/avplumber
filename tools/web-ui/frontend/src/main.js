import App from './App.svelte';

// Golden Layout styles (base + dark theme). Our local overrides are in styles.css.
import 'golden-layout/dist/css/goldenlayout-base.css';
import 'golden-layout/dist/css/themes/goldenlayout-dark-theme.css';
import './styles.css';

const app = new App({
  target: document.getElementById('app')
});

export default app;



