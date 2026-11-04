#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 gpu-monitor contributors
/**
 * Render the real dashboard headlessly against synthetic samples.
 *
 * The README's screenshots are claims about the UI, so they are generated
 * from the shipped `app/static/index.html` rather than photographed from
 * someone's desktop: the same HTML, the same bundled Chart.js, the same
 * WebSocket protocol — only the metrics are fabricated, and only because no
 * CI machine has an Intel iGPU, an AMD card and an NVIDIA card in it.
 *
 * It also doubles as the regression test for the single-screen layout:
 * `--check` asserts that nothing overflows the viewport for one, two or
 * three GPUs at the sizes the layout promises to fit.
 *
 *   node tools/screenshots.mjs                 # regenerate docs/*.png
 *   node tools/screenshots.mjs --check         # no-scroll assertions only
 *   node tools/screenshots.mjs --gpus intel    # one-off render, custom set
 *
 * Requires Playwright with Chromium (`npm i -g playwright && playwright
 * install chromium`). Nothing else — no FastAPI, no GPU, no network.
 */

import { chromium } from './playwright-loader.mjs';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const STATIC_DIR = path.join(ROOT, 'app', 'static');
const DOCS_DIR = path.join(ROOT, 'docs');
const ORIGIN = 'http://gpu-monitor.test';

const WINDOW_S = 60;
const INTERVAL_S = 1;
const POINTS = WINDOW_S / INTERVAL_S + 1;

/* ── deterministic noise ──────────────────────────────────────────────────
   A seeded PRNG, so re-running this produces byte-identical screenshots and
   a diff in docs/ means the UI changed, not that the dice did. */
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** A smooth bounded walk — engine counters wander, they don't teleport. */
function walk(rand, { start, min, max, step }) {
  const out = [];
  let v = start;
  let drift = 0;
  for (let i = 0; i < POINTS; i++) {
    drift = drift * 0.72 + (rand() - 0.5) * step;
    v = Math.min(max, Math.max(min, v + drift));
    out.push(Math.round(v * 10) / 10);
  }
  return out;
}

/* ── synthetic hosts ──────────────────────────────────────────────────────
   Each entry mirrors exactly what its backend can actually report. The Intel
   iGPU has no VRAM, no power sensor and no temperature sensor exposed through
   i915, so those keys are absent — the dashboard must render "—" for them,
   and the screenshot has to show that rather than a fabricated zero. */
const HOSTS = {
  intel: {
    uid: 'intel:0000:00:02.0',
    vendor: 'intel',
    name: 'Intel UHD Graphics 630',
    pci_address: '0000:00:02.0',
    driver: 'i915',
    engine_names: ['Render/3D', 'Blitter', 'Video', 'VideoEnhance'],
    engine_keys: ['render', 'blitter', 'video', 'video_enhance'],
    engines: [
      { start: 11, min: 3, max: 28, step: 3 },
      { start: 1.4, min: 0, max: 6, step: 1.2 },
      { start: 58, min: 34, max: 78, step: 6 },
      { start: 22, min: 8, max: 40, step: 4 }
    ],
    frequency: { max: 1150, walk: { start: 833, min: 350, max: 1150, step: 90 } },
    rc6: { start: 14, min: 4, max: 46, step: 7 },
    containers: {
      jellyfin: 0.86,
      frigate: 0.09,
      'immich-machine-learning': 0,
      plex: 0,
      host: 0.02
    }
  },

  amd: {
    uid: 'amd:0000:03:00.0',
    vendor: 'amd',
    name: 'AMD Radeon RX 7600',
    pci_address: '0000:03:00.0',
    driver: 'amdgpu',
    engine_names: ['Graphics', 'Compute', 'Decode', 'Encode'],
    engine_keys: ['graphics', 'compute', 'decode', 'encode'],
    engines: [
      { start: 49, min: 22, max: 72, step: 5 },
      { start: 66, min: 38, max: 88, step: 6 },
      { start: 8, min: 0, max: 18, step: 2.5 },
      { start: 11, min: 2, max: 24, step: 3 }
    ],
    frequency: { max: 2755, walk: { start: 1686, min: 500, max: 2755, step: 160 } },
    memory: { total: 8192, walk: { start: 3820, min: 2600, max: 6900, step: 210 }, bus: { start: 25, min: 8, max: 52, step: 5 } },
    power: { start: 105, min: 42, max: 158, step: 9 },
    powerCap: 165,
    temp: { start: 63, min: 52, max: 74, step: 1.4 },
    /* amdgpu exports edge, junction (hotspot) and VRAM temperatures, a fan
       and vddgfx — the same set `sensors` prints for the card. */
    sensors: {
      temperatures: [
        { key: 'edge', label: 'Edge' },
        { key: 'junction', label: 'Hotspot' },
        { key: 'mem', label: 'VRAM' }
      ],
      fan: true,
      voltage: true
    },
    hotspot: { start: 74, min: 61, max: 88, step: 1.8 },
    memtemp: { start: 68, min: 58, max: 80, step: 1.2 },
    fan: { start: 1450, min: 900, max: 2100, step: 90 },
    containers: { ollama: 0.7, 'stable-diffusion': 0.24, handbrake: 0.06, host: 0 }
  },

  nvidia: {
    uid: 'nvidia:GPU-6f2b1c48-2b73-4d2a-9b41-2a7f5c0d9e11',
    vendor: 'nvidia',
    name: 'NVIDIA GeForce RTX 3060',
    pci_address: '0000:09:00.0',
    driver: 'nvidia',
    engine_names: ['Graphics', 'Decode', 'Encode'],
    engine_keys: ['graphics', 'decode', 'encode'],
    engines: [
      { start: 53, min: 24, max: 79, step: 5 },
      { start: 12, min: 2, max: 26, step: 3 },
      { start: 38, min: 14, max: 58, step: 4.5 }
    ],
    frequency: { max: 1852, walk: { start: 1531, min: 420, max: 1852, step: 120 } },
    memory: { total: 12288, walk: { start: 5275, min: 3900, max: 9200, step: 240 } },
    power: { start: 102, min: 30, max: 168, step: 11 },
    temp: { start: 56, min: 44, max: 68, step: 1.2 },
    containers: { ollama: 0.52, jellyfin: 0.45, 'tdarr-node': 0, host: 0.03 }
  }
};

/** Build the `devices` frame plus a full window of `history` for one host. */
function synthesise(names) {
  const devices = [];
  const series = [];

  const seen = new Map();

  names.forEach((key, index) => {
    const template = HOSTS[key];
    if (!template) throw new Error(`unknown GPU '${key}' (have: ${Object.keys(HOSTS).join(', ')})`);
    const rand = rng(0x51de0000 + index * 977);

    /* Repeating a vendor is how the 4+ GPU layouts get exercised, and uid
       has to stay unique — the dashboard keys every card, dataset and row on
       it, so duplicates would leave the second card frozen. */
    const dupe = seen.get(key) || 0;
    seen.set(key, dupe + 1);
    const host = dupe ? { ...template, uid: `${template.uid}#${dupe}` } : template;

    devices.push({
      uid: host.uid,
      vendor: host.vendor,
      name: host.name,
      index,
      pci_address: host.pci_address,
      driver: host.driver,
      engine_names: host.engine_names,
      engine_keys: host.engine_keys,
      sensors: host.sensors || {},
      backend_error: null
    });

    series.push({
      host,
      engines: host.engines.map((spec) => walk(rand, spec)),
      freq: walk(rand, host.frequency.walk),
      rc6: host.rc6 ? walk(rand, host.rc6) : null,
      mem: host.memory ? walk(rand, host.memory.walk) : null,
      membus: host.memory && host.memory.bus ? walk(rand, host.memory.bus) : null,
      power: host.power ? walk(rand, host.power) : null,
      temp: host.temp ? walk(rand, host.temp) : null,
      hotspot: host.hotspot ? walk(rand, host.hotspot) : null,
      memtemp: host.memtemp ? walk(rand, host.memtemp) : null,
      fan: host.fan ? walk(rand, host.fan) : null
    });
  });

  const frames = [];
  for (let t = 0; t < POINTS; t++) {
    const gpus = series.map((s, i) => {
      const host = s.host;
      const engines = {};
      host.engine_names.forEach((name, e) => { engines[name] = { busy: s.engines[e][t] }; });
      const busiest = Math.max(...host.engine_names.map((n) => engines[n].busy));

      const block = {
        uid: host.uid,
        vendor: host.vendor,
        name: host.name,
        index: i,
        engines,
        utilisation: { value: busiest },
        frequency: { actual: s.freq[t], requested: s.freq[t], max: host.frequency.max }
      };

      if (s.rc6) block.rc6 = { value: s.rc6[t] };
      if (s.mem) {
        block.memory = {
          used_mib: s.mem[t],
          total_mib: host.memory.total,
          used_pct: Math.round((s.mem[t] / host.memory.total) * 1000) / 10
        };
        if (s.membus) block.memory.busy_pct = s.membus[t];
      }
      if (s.power) {
        block.power = { watts: s.power[t] };
        if (host.powerCap) block.power.cap_watts = host.powerCap;
      }
      if (s.temp) block.temperature = { celsius: s.temp[t] };
      if (s.temp && s.hotspot) {
        block.temperature.sensor = 'edge';
        block.temperature.channels = { edge: s.temp[t], junction: s.hotspot[t] };
        if (s.memtemp) block.temperature.channels.mem = s.memtemp[t];
      }
      if (s.fan) block.fan = { rpm: s.fan[t] };

      /* Attribution splits the busy total across the named containers in the
         host's fixed shares, so the bars stay consistent with the plot above
         them instead of telling a different story. */
      const containers = {};
      const totalBusy = host.engine_names.reduce((sum, n) => sum + engines[n].busy, 0);
      Object.entries(host.containers).forEach(([name, share]) => {
        const total = Math.round(totalBusy * share * 10) / 10;
        const bucket = { total };
        host.engine_keys.forEach((k, e) => {
          bucket[k] = Math.round(s.engines[e][t] * share * 10) / 10;
        });
        containers[name] = bucket;
      });
      block.containers = containers;

      return block;
    });

    frames.push({
      timestamp: new Date(Date.UTC(2026, 6, 30, 9, 41, 0) + t * 1000).toISOString(),
      gpus
    });
  }

  const config = {
    vendors: { intel: 'auto', amd: 'auto', nvidia: 'auto' },
    sample_interval_s: INTERVAL_S,
    history_size: 300,
    container_attribution: true,
    chart_window_s: WINDOW_S,
    legacy_sample_fields: true
  };

  return {
    devices: { gpus: devices, config },
    history: frames,
    live: [frames[frames.length - 1]],
    health: {
      status: 'ok',
      gpu_detected: true,
      gpu_count: devices.length,
      device_present: true,
      has_data: true,
      active_source: 'synthetic',
      gpu_error: null,
      gpu_error_kind: null,
      backends: [],
      config,
      container_resolver: { available: true, degraded: false }
    }
  };
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml'
};

/**
 * Open the dashboard with a fake `/ws` and `/health`.
 *
 * The page is served over http (not file://) so `location.host`, the
 * WebSocket URL and the `/health` poll all behave exactly as they do in
 * production; only the socket implementation is swapped, before any page
 * script runs.
 */
async function openDashboard(browser, frames, viewport) {
  const context = await browser.newContext({
    viewport,
    deviceScaleFactor: 1.5,
    /* Deterministic captures: the entrance fade and the bar transitions are
       real, but a screenshot should not depend on catching them mid-flight. */
    reducedMotion: 'reduce'
  });

  const page = await context.newPage();

  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === '/health') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(frames.health)
      });
    }
    const rel = url.pathname === '/' ? '/index.html' : url.pathname;
    const file = path.join(STATIC_DIR, path.normalize(rel));
    if (!file.startsWith(STATIC_DIR)) return route.abort();
    try {
      const body = await readFile(file);
      return route.fulfill({
        status: 200,
        contentType: MIME[path.extname(file)] || 'application/octet-stream',
        body
      });
    } catch {
      return route.fulfill({ status: 404, body: 'not found' });
    }
  });

  await page.addInitScript(({ devices, history, live }) => {
    class FakeSocket {
      constructor(url) {
        this.url = url;
        this.readyState = 0;
        setTimeout(() => {
          this.readyState = 1;
          if (this.onopen) this.onopen({});
          this._emit({ type: 'devices', data: devices });
          this._emit({ type: 'history', data: history });
          let i = 0;
          this._timer = setInterval(() => {
            if (i >= live.length) { clearInterval(this._timer); return; }
            this._emit({ type: 'sample', data: live[i++] });
          }, 100);
        }, 10);
      }
      _emit(obj) { if (this.onmessage) this.onmessage({ data: JSON.stringify(obj) }); }
      send() {}
      close() { clearInterval(this._timer); this.readyState = 3; }
    }
    window.WebSocket = FakeSocket;
  }, { devices: frames.devices, history: frames.history, live: frames.live });

  await page.goto(`${ORIGIN}/`, { waitUntil: 'load' });
  await page.waitForSelector('.gpu .bar-row', { timeout: 10000 });
  await page.waitForFunction(() => {
    const dot = document.getElementById('dot');
    return dot && dot.classList.contains('live');
  });
  /* One extra beat so the final live frame has landed and the ResizeObserver
     pass has settled the x-axis density and the row cap. */
  await page.waitForTimeout(450);

  return { context, page };
}

/* ── overflow assertions ─────────────────────────────────────────────────
   The promise the redesign makes: at 1280x800 and up, three GPUs fit with no
   page scrollbar. These sizes are the floor, one 16:10 laptop, one 16:9
   laptop and 1080p. */
const CHECK_VIEWPORTS = [
  { width: 1280, height: 800 },
  { width: 1366, height: 768 },
  { width: 1920, height: 1080 }
];
const CHECK_SETS = [['intel'], ['intel', 'amd'], ['intel', 'amd', 'nvidia']];

/* Beyond the promise. The rule the grid follows is "one column per GPU until
   a column would be narrower than 330px, then wrap" — so four cards on a
   1080p screen stay in one row, and six on a laptop wrap to two rows of
   three and shed their sparklines. Both must still fit. The phone is allowed
   to scroll vertically; it is not allowed to scroll sideways. */
const EXTRA_CASES = [
  { set: ['intel', 'amd', 'nvidia', 'amd'], viewport: { width: 1920, height: 1080 }, rows: 1 },
  { set: ['intel', 'amd', 'nvidia', 'amd', 'nvidia', 'intel'], viewport: { width: 1280, height: 800 }, rows: 2 },
  { set: ['intel', 'amd', 'nvidia'], viewport: { width: 390, height: 844 }, rows: 3, scrolls: true }
];

async function runChecks(browser) {
  let failures = 0;
  const cases = [];
  for (const set of CHECK_SETS) {
    for (const viewport of CHECK_VIEWPORTS) cases.push({ set, viewport, rows: 1 });
  }
  cases.push(...EXTRA_CASES);

  for (const testCase of cases) {
    {
      const { set, viewport } = testCase;
      const frames = synthesise(set);
      const { context, page } = await openDashboard(browser, frames, viewport);
      const report = await page.evaluate(() => {
        const doc = document.documentElement;
        const cards = Array.from(document.querySelectorAll('.gpu'));
        const tops = new Set(cards.map((c) => Math.round(c.getBoundingClientRect().top)));
        return {
          scrollH: doc.scrollHeight,
          clientH: doc.clientHeight,
          scrollW: doc.scrollWidth,
          clientW: doc.clientWidth,
          rows: tops.size,
          cards: cards.length,
          clipped: cards.some((c) => {
            const plot = c.querySelector('.plot');
            return !plot || plot.clientHeight < 60;
          })
        };
      });
      await context.close();

      const problems = [];
      if (!testCase.scrolls && report.scrollH > report.clientH + 1) {
        problems.push(`vertical overflow (${report.scrollH} > ${report.clientH})`);
      }
      if (report.scrollW > report.clientW + 1) {
        problems.push(`horizontal overflow (${report.scrollW} > ${report.clientW})`);
      }
      if (report.cards !== set.length) {
        problems.push(`rendered ${report.cards} cards, expected ${set.length}`);
      }
      if (report.rows !== testCase.rows) {
        problems.push(`cards occupy ${report.rows} rows, expected ${testCase.rows}`);
      }
      if (report.clipped) problems.push('a plot collapsed below 60px');

      const label = `${String(set.length).padStart(2)} GPU${set.length === 1 ? ' ' : 's'} @ ${viewport.width}x${viewport.height}`.padEnd(24);
      if (problems.length) {
        failures++;
        console.error(`  FAIL  ${label} ${problems.join('; ')}`);
      } else {
        const fit = testCase.scrolls ? 'scrolls (as intended)' : 'fits';
        console.log(`  ok    ${label} ${fit}, ${report.rows} row(s), plots intact`);
      }
    }
  }

  return failures;
}

/* ── entry point ─────────────────────────────────────────────────────── */

function parseArgs(argv) {
  const args = { check: false, gpus: null, out: null, width: 1600, height: 900 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--check') args.check = true;
    else if (a === '--gpus') args.gpus = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (a === '--out') args.out = argv[++i];
    else if (a === '--width') args.width = Number(argv[++i]);
    else if (a === '--height') args.height = Number(argv[++i]);
    else throw new Error(`unknown argument '${a}'`);
  }
  return args;
}

async function shoot(browser, set, out, viewport) {
  const { context, page } = await openDashboard(browser, synthesise(set), viewport);
  await page.screenshot({ path: out });
  await context.close();
  console.log(`  wrote ${path.relative(ROOT, out)}  (${set.join(' + ')} @ ${viewport.width}x${viewport.height})`);
}

const args = parseArgs(process.argv.slice(2));
const browser = await chromium.launch();
let exit = 0;

try {
  if (args.check) {
    console.log('Single-screen layout checks:');
    exit = (await runChecks(browser)) === 0 ? 0 : 1;
    console.log(exit === 0 ? 'All viewports fit without scrolling.' : 'Layout checks FAILED.');
  } else if (args.gpus) {
    const out = path.resolve(ROOT, args.out || path.join(DOCS_DIR, `screenshot-${args.gpus.join('-')}.png`));
    await shoot(browser, args.gpus, out, { width: args.width, height: args.height });
  } else {
    console.log('Rendering README screenshots:');
    const viewport = { width: args.width, height: args.height };
    await shoot(browser, ['intel', 'amd', 'nvidia'], path.join(DOCS_DIR, 'screenshot.png'), viewport);
    await shoot(browser, ['intel'], path.join(DOCS_DIR, 'screenshot-intel.png'), viewport);
  }
} finally {
  await browser.close();
}

process.exit(exit);
