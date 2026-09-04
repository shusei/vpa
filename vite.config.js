import { cpSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

const root = import.meta.dirname;
const outDir = resolve(root, "dist");
const packageVersion = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8")).version;

function copyRuntimeAssets() {
  const directories = ["audio", "data", "img", "vendor"];
  const files = [".nojekyll", "avatar-evelyn.jpg", "avatar-shusei.jpg", "favicon.ico", "ogp.png"];
  return {
    name: "vpa-copy-runtime-assets",
    closeBundle() {
      mkdirSync(resolve(outDir, "assets"), { recursive: true });
      directories.forEach((name) => {
        const source = resolve(root, "assets", name);
        if (existsSync(source)) cpSync(source, resolve(outDir, "assets", name), { recursive: true });
      });
      files.forEach((name) => {
        const source = name.startsWith("avatar-")
          ? resolve(root, "assets", name)
          : resolve(root, name);
        const destination = name.startsWith("avatar-")
          ? resolve(outDir, "assets", name)
          : resolve(outDir, name);
        if (existsSync(source)) cpSync(source, destination);
      });
    },
  };
}

export default defineConfig({
  base: "/vpa/",
  define: {
    __VPA_BUILD_VERSION__: JSON.stringify(packageVersion),
  },
  publicDir: false,
  resolve: {
    alias: {
      "@ffmpeg/ffmpeg": resolve(root, "assets/vendor/ffmpeg/index.js"),
    },
  },
  build: {
    assetsDir: "assets/build",
    emptyOutDir: true,
    outDir,
    target: "es2022",
    rollupOptions: {
      input: {
        main: resolve(root, "index.html"),
        phrases: resolve(root, "assets/ui/phrases-lab.html"),
      },
      output: {
        assetFileNames: "assets/build/[name]-[hash][extname]",
        chunkFileNames: "assets/build/[name]-[hash].js",
        entryFileNames: "assets/build/[name]-[hash].js",
      },
    },
  },
  plugins: [copyRuntimeAssets()],
});
