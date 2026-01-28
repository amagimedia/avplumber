import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import sveltePreprocess from 'svelte-preprocess';

export default defineConfig({
  plugins: [
    svelte({
      preprocess: sveltePreprocess({
        typescript: {
          tsconfigFile: './tsconfig.json',
          compilerOptions: {
            verbatimModuleSyntax: true
          }
        }
      })
    })
  ],
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
});


