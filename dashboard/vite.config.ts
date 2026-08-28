import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://localhost:8000',
      '/predict': 'http://localhost:8000',
      '/cases': 'http://localhost:8000',
      '/reports': 'http://localhost:8000',
      '/outputs': 'http://localhost:8000',
    },
  },
})
