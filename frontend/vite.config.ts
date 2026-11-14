/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend, so the browser talks to a single origin.
const backend = process.env.SONICSPHERE_BACKEND ?? "http://127.0.0.1:8000";

export default defineConfig({
  // GitHub Pages serves project sites from /<repo-name>/; the deploy workflow sets VITE_BASE.
  base: process.env.VITE_BASE ?? "/",
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": { target: backend, changeOrigin: true } } },
  preview: { port: 4173, proxy: { "/api": { target: backend, changeOrigin: true } } },
  worker: { format: "es" },
  build: { target: "es2020", chunkSizeWarningLimit: 1600 },
  test: { environment: "node", include: ["src/**/*.test.ts"], testTimeout: 60000 },
});
