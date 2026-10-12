import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Requests go to the same origin (/upload/documents, /search) and are proxied to the
// FastAPI server, so the browser never makes a cross-origin request and the backend
// needs no CORS configuration.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/upload": "http://127.0.0.1:8000",
      "/search": "http://127.0.0.1:8000",
    },
  },
});
