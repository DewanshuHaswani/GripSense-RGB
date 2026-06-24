import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const inferenceTarget = process.env.VITE_GRIPSENSE_INFERENCE_TARGET ?? 'http://127.0.0.1:7867';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api/gripsense/rfdetr/analyze': {
        target: inferenceTarget,
        changeOrigin: true,
        secure: false,
        rewrite: () => '/api/rfdetr/analyze'
      },
      '/api/gripsense/yolo/analyze': {
        target: inferenceTarget,
        changeOrigin: true,
        secure: false,
        rewrite: () => '/api/yolo/analyze'
      },
      '/api/gripsense/realsense/depth-signal': {
        target: inferenceTarget,
        changeOrigin: true,
        secure: false,
        rewrite: () => '/api/realsense/depth-signal'
      },
      '/api/gripsense/v3/analyze-frame': {
        target: inferenceTarget,
        changeOrigin: true,
        secure: false,
        rewrite: () => '/v3/analyze-frame'
      }
    }
  }
});
