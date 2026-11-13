/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  // Deploy targets with a base path (GitHub Pages under /autoquant/) must
  // pass --base=/autoquant/ to `vite build`; default build is root-relative
  // for Vercel.
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        // Keep the entry (landing) chunk small (#39) by splitting the
        // generated snapshot data it shares; node_modules are left to
        // Rollup's default splitting so table/router/icon code is not
        // forced eager for the landing page.
        manualChunks(id) {
          if (id.includes("snapshot.json")) return "snapshot";
          if (id.includes("workspace.json")) return "workspace";
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./tests/setup.ts",
    globals: true,
  },
});
