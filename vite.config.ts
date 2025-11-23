import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  root: '.',
  server: {
    port: 5183
  },
  build: {
    rollupOptions: {
      input: resolve(__dirname, 'index.html')
    }
  },
  plugins: [react()]
})

