import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  // If you had a 'server' block with a 'proxy' section pointing to 
  // 'http://localhost:8000', it has been REMOVED here. 
  // You no longer need proxying since the backend is deployed.
  server: {
    // If you had a proxy block like this, make sure it is GONE:
    // proxy: {
    //   '/api': {
    //     target: 'http://localhost:8000', 
    //     changeOrigin: true,
    //     rewrite: (path) => path.replace(/^\/api/, ''),
    //   },
    // },
  },
});
