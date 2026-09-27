import { defineConfig } from "vite";

// The API serves this directory in production; Vite is intentionally build-only.
export default defineConfig({
  base: "/verify/",
  publicDir: false,
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
