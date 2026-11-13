// Publish web/dist as a public web page (GitHub Pages) by pushing it to the repo's
// "gh-pages" branch. Run:  node deploy.mjs
//
// - The branch holds only the latest build (it is replaced each time), so the large
//   model and library files never pile up in the repo's history.
// - Phones need https for the camera, which the published page has and a local server lacks.
// - The page is PUBLIC: anyone with the link can open it. It holds no patient data,
//   only the app and two fictional sample documents.

import { execFileSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const web = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(web), dist = path.join(web, "dist");
const git = (args, cwd = repo) => execFileSync("git", args, { cwd }).toString().trim();

const origin = git(["remote", "get-url", "origin"]);
const name = git(["config", "user.name"]), email = git(["config", "user.email"]);
if (!email.endsWith("@users.noreply.github.com")) {
  throw new Error(`This repo's git email (${email}) is not a GitHub no-reply address: not publishing with it.`);
}

await import("./build.mjs");   // a fresh build (also removes anything left in dist)

rmSync(path.join(dist, ".git"), { recursive: true, force: true });
git(["init", "-q", "-b", "gh-pages"], dist);
git(["add", "-A"], dist);
git(["-c", `user.name=${name}`, "-c", `user.email=${email}`, "commit", "-q", "-m",
     process.env.DEPLOY_MESSAGE || "Publish the MedScan web trial"], dist);
execFileSync("git", ["push", "-f", origin, "gh-pages"], { cwd: dist, stdio: "inherit" });
rmSync(path.join(dist, ".git"), { recursive: true, force: true });

const m = origin.match(/github\.com[:/]([^/]+)\/(.+?)(\.git)?$/);
if (m) console.log(`\nPublished. Page: https://${m[1]}.github.io/${m[2]}/\n` +
  "First time only: on GitHub, Settings > Pages > Source: 'Deploy from a branch', branch 'gh-pages', folder '/ (root)'.");
