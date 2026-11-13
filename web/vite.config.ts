import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development the page runs on Vite's port and the API on 8000 (`make serve`).
// The proxy sends API calls across, so the browser only ever talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ask': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
