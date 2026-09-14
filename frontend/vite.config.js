import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发时后端代理到 FastAPI(8000)；构建产物由 FastAPI 挂载到 /
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/data': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
  build: {
    outDir: 'dist',
  },
})
