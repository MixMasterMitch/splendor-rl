import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

function resolvePlayServer(): string {
  return process.env["PLAY_SERVER"] ?? "http://127.0.0.1:8765";
}

// ---------------------------------------------------------------------------
// Vite config
// ---------------------------------------------------------------------------
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: false,
    proxy: {
      "/api/play": {
        target: resolvePlayServer(),
        changeOrigin: true,
        rewrite: (p: string) => p.replace(/^\/api\/play/, "/api"),
      },
    },
  },
});
