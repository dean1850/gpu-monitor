// SPDX-License-Identifier: MIT
// Copyright (c) 2026 gpu-monitor contributors
/**
 * Import Playwright from wherever it happens to live.
 *
 * This repository has no package.json and no npm dependency tree — it is a
 * Python service with one vendored JS file. Screenshot generation is a
 * developer-only errand, so Playwright is expected to be installed globally
 * (`npm i -g playwright`), and a bare `import 'playwright'` does not resolve
 * global installs. Rather than add a package.json purely to satisfy the
 * resolver, try the obvious locations and fail with an actionable message.
 */

import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

async function attempt(specifier) {
  try {
    return await import(specifier);
  } catch {
    return null;
  }
}

function globalRoot() {
  try {
    return execFileSync('npm', ['root', '-g'], { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

async function load() {
  const local = await attempt('playwright');
  if (local) return local;

  const roots = [globalRoot(), '/usr/lib/node_modules', '/usr/local/lib/node_modules']
    .filter(Boolean);

  for (const root of roots) {
    const entry = pathToFileURL(path.join(root, 'playwright', 'index.mjs')).href;
    const found = await attempt(entry);
    if (found) return found;
    const cjs = pathToFileURL(path.join(root, 'playwright', 'index.js')).href;
    const foundCjs = await attempt(cjs);
    if (foundCjs) return foundCjs.default || foundCjs;
  }

  throw new Error(
    'Playwright not found. Install it and its Chromium build:\n' +
    '    npm i -g playwright && playwright install chromium\n' +
    '(or `npm i -D playwright` inside this checkout).'
  );
}

const playwright = await load();
export const chromium = playwright.chromium;
