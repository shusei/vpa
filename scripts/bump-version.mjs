import fs from "node:fs";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dirname, "..");
const pkgPath = path.resolve(projectRoot, "package.json");
const lockPath = path.resolve(projectRoot, "package-lock.json");
const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));

const currentVersion = pkg.version;
const type = process.argv[2] || "patch";
const parts = currentVersion.split(".").map(Number);

if (type === "minor") {
  parts[1]++;
  parts[2] = 0;
} else if (type === "major") {
  parts[0]++;
  parts[1] = 0;
  parts[2] = 0;
} else {
  parts[2]++;
}

const nextVersion = parts.join(".");
pkg.version = nextVersion;
fs.writeFileSync(pkgPath, `${JSON.stringify(pkg, null, 2)}\n`, "utf8");

if (fs.existsSync(lockPath)) {
  const lock = JSON.parse(fs.readFileSync(lockPath, "utf8"));
  lock.version = nextVersion;
  if (lock.packages?.[""]) lock.packages[""].version = nextVersion;
  fs.writeFileSync(lockPath, `${JSON.stringify(lock, null, 2)}\n`, "utf8");
}

console.log(`=== Bumping Version: ${currentVersion} -> ${nextVersion} ===`);

// Production deploys use Vite content hashes. These source tags remain useful for
// direct, no-build development, so discover them recursively instead of relying
// on a hand-maintained file list that can miss newly added modules.
const cacheTagExtensions = new Set([".css", ".html", ".js", ".mjs"]);
const cacheTagRoots = ["index.html", "dev.html", "assets", "tests/verify-social-preview.mjs"];

function collectCacheTagFiles(relativePath) {
  const absolutePath = path.resolve(projectRoot, relativePath);
  if (!fs.existsSync(absolutePath)) return [];
  const stat = fs.statSync(absolutePath);
  if (stat.isFile()) {
    return cacheTagExtensions.has(path.extname(absolutePath)) ? [relativePath] : [];
  }
  return fs.readdirSync(absolutePath, { withFileTypes: true })
    .flatMap((entry) => collectCacheTagFiles(path.join(relativePath, entry.name)));
}

const newVersionTag = nextVersion;
const filesToUpdate = cacheTagRoots.flatMap(collectCacheTagFiles);

for (const relativePath of filesToUpdate) {
  const absolutePath = path.resolve(projectRoot, relativePath);
  const content = fs.readFileSync(absolutePath, "utf8");
  const updated = content.replace(/\?v=[a-zA-Z0-9.-]+/g, `?v=${newVersionTag}`);
  if (updated === content) continue;
  fs.writeFileSync(absolutePath, updated, "utf8");
  console.log(`Cache-busting updated: ${relativePath}`);
}

console.log(`\nSuccessfully bumped version to ${nextVersion}; production filenames are content-hashed by Vite.\n`);
