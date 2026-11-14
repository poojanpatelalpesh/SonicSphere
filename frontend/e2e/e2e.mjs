import { chromium } from "playwright-core";
// End-to-end test of the running app in a real browser (dev server + backend must be running).
//   npm i --no-save playwright-core@1.48.2   (in frontend/)
//   node e2e/e2e.mjs <screenshot-dir>          (env CHROME=path/to/chrome to override)
import { mkdirSync } from "node:fs";
const OUT = process.argv[2] ?? "e2e-shots";
mkdirSync(OUT, { recursive: true });
const CHROME = process.env.CHROME ?? (process.platform === "win32" ? "C:/Program Files/Google/Chrome/Application/chrome.exe" : undefined);
const browser = await chromium.launch({ executablePath: CHROME, channel: CHROME ? undefined : "chrome", headless: true,
  args: ["--autoplay-policy=no-user-gesture-required", "--use-gl=angle", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(`${m.type()}: ${m.text()}`); });
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
await page.goto("http://localhost:5173/", { waitUntil: "load", timeout: 60000 });
await page.waitForFunction(() => window.__sonicsphere?.useStore.getState().sounds.filter((s) => s.peaks).length >= 4, null, { timeout: 30000 });
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}/01_initial_dark.png` });
const res = {};
await page.getByRole("button", { name: "Put on headphones, then start sound" }).click();
await page.waitForFunction(() => window.__sonicsphere.useStore.getState().audioState === "running", null, { timeout: 20000 });
res.start = await page.evaluate(() => { const s = window.__sonicsphere.useStore.getState(); return { sr: s.sampleRate, method: s.method, bench: s.bench?.reason, timer: s.timerInWorklet, base: s.baseLatency, out: s.outputLatency }; });
await page.getByRole("button", { name: "Play", exact: true }).click();
await page.waitForFunction(() => window.__sonicsphere.liveRef.status?.transport.playing, null, { timeout: 10000 });

// Energy at the two ears for sources placed via the store (same path as click/drag).
const measure = async (az, el) => page.evaluate(async ([az, el]) => {
  const { useStore, getEngine } = window.__sonicsphere;
  const r = 1.5, a = az * Math.PI / 180, e = el * Math.PI / 180;
  const h = [r * Math.cos(e) * Math.cos(a), r * Math.cos(e) * Math.sin(a), r * Math.sin(e)];
  useStore.getState().setSourcePos([0 - h[1], 1.15 + h[2], 0 - h[0]]);
  await new Promise((f) => setTimeout(f, 600));
  const eng = getEngine(); const bl = new Float32Array(2048), br = new Float32Array(2048);
  let el2 = 0, er2 = 0;
  for (let k = 0; k < 25; k++) { eng.analyserL.getFloatTimeDomainData(bl); eng.analyserR.getFloatTimeDomainData(br);
    for (let i = 0; i < 2048; i++) { el2 += bl[i] * bl[i]; er2 += br[i] * br[i]; } await new Promise((f) => setTimeout(f, 40)); }
  const v = window.__sonicsphere.liveRef.status.voices[0];
  return { az, el, ild_db: 10 * Math.log10(el2 / er2), rms_db: 10 * Math.log10((el2 + er2) / (2 * 25 * 2048)), status: v.status, qaz: v.az, qel: v.el };
}, [az, el]);
res.static = [];
for (const [az, el] of [[90, 0], [270, 0], [0, 0], [180, 0], [37, 12], [90, -40]]) res.static.push(await measure(az, el));
await page.screenshot({ path: `${OUT}/02_playing.png` });
// Broadband test signal through the real upload path (stereo WAV, identical channels).
res.upload = await page.evaluate(async () => {
  const fs = 44100, n = fs * 3, ch = 2;
  const buf = new ArrayBuffer(44 + n * ch * 2), dv = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => dv.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); dv.setUint32(4, 36 + n * ch * 2, true); w(8, "WAVEfmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
  dv.setUint16(22, ch, true); dv.setUint32(24, fs, true); dv.setUint32(28, fs * ch * 2, true); dv.setUint16(32, ch * 2, true); dv.setUint16(34, 16, true);
  w(36, "data"); dv.setUint32(40, n * ch * 2, true);
  for (let i = 0; i < n; i++) { const v = Math.round((Math.random() * 2 - 1) * 9000); dv.setInt16(44 + 4 * i, v, true); dv.setInt16(46 + 4 * i, v, true); }
  const st = window.__sonicsphere.useStore.getState();
  await st.addUpload(new File([buf], "noise.wav", { type: "audio/wav" }));
  await st.play();
  const s2 = window.__sonicsphere.useStore.getState();
  return { selected: s2.selectedSoundId, policy: s2.prepared?.policy, analysis: s2.prepared?.analysis, sounds: s2.sounds.length };
});
res.noise = [];
// Direct-sound cues measured without the (direction-independent) room ambience.
await page.evaluate(() => window.__sonicsphere.useStore.getState().setAmbience(false));
for (const [az, el] of [[90, 0], [270, 0], [0, 0], [45, 0], [315, 0]]) res.noise.push(await measure(az, el));
// 3D off: both ears must get the same signal.
await page.evaluate(() => window.__sonicsphere.useStore.getState().setSpatial(false));
res.bypass = await measure(90, 0);
await page.evaluate(() => window.__sonicsphere.useStore.getState().setSpatial(true));
// With ambience on, the side is still clearly audible.
await page.evaluate(() => window.__sonicsphere.useStore.getState().setAmbience(true));
res.ambience = await measure(90, 0);
// Headphone check: left-only bursts must arrive in the left channel only.
res.check = await page.evaluate(async () => {
  const { useStore, getEngine } = window.__sonicsphere;
  await useStore.getState().checkHeadphones("left");
  await new Promise((f) => setTimeout(f, 200));
  const eng = getEngine(), bl = new Float32Array(2048), br = new Float32Array(2048);
  let l = 0, r = 0;
  for (let k = 0; k < 15; k++) {
    eng.analyserL.getFloatTimeDomainData(bl); eng.analyserR.getFloatTimeDomainData(br);
    for (let i = 0; i < 2048; i++) { l += bl[i] * bl[i]; r += br[i] * br[i]; }
    await new Promise((f) => setTimeout(f, 40));
  }
  await new Promise((f) => setTimeout(f, 900));
  await useStore.getState().play();
  return { left: l, right: r, ratio_db: 10 * Math.log10((l + 1e-12) / (r + 1e-12)) };
});
// Invalid upload must be rejected with a message.
res.badUpload = await page.evaluate(async () => {
  const st = window.__sonicsphere.useStore.getState();
  await st.addUpload(new File([new Uint8Array(4000).fill(7)], "broken.wav", { type: "audio/wav" }));
  return window.__sonicsphere.useStore.getState().toast;
});
await page.evaluate(async () => { const st = window.__sonicsphere.useStore.getState(); await st.selectSound("helicopter"); await st.play(); });

// Trajectory: orbit preset, sample the direction reported by the audio thread.
await page.getByRole("button", { name: "Orbit" }).click();
await page.getByRole("button", { name: "Play path" }).click();
res.traj = await page.evaluate(async () => {
  const out = [];
  for (let k = 0; k < 30; k++) { const s = window.__sonicsphere.liveRef.status; out.push([+s.trajectory.phase.toFixed(3), +s.voices[0].az.toFixed(1), s.stats.crossfades]); await new Promise((f) => setTimeout(f, 150)); }
  const s = window.__sonicsphere.liveRef.status;
  return { samples: out, timing: s.timing, cache: s.stats, limiter: s.limiterMinGain, playing: s.transport.playing };
});
await page.screenshot({ path: `${OUT}/03_trajectory.png` });
// Switch convolution method & interpolation while moving – must keep running.
res.switch = await page.evaluate(async () => {
  const st = window.__sonicsphere.useStore.getState();
  st.setMethodChoice("direct"); await new Promise((f) => setTimeout(f, 500));
  const a = window.__sonicsphere.liveRef.status.method;
  st.setMethodChoice("partitioned"); st.setInterp("nearest"); await new Promise((f) => setTimeout(f, 500));
  const b = window.__sonicsphere.liveRef.status;
  st.setInterp("barycentric_aligned"); st.setMethodChoice("auto");
  return { direct: a, after: b.method, interp: b.interp, playing: b.transport.playing, frame: b.frame };
});
// "Try this" experience: whisper in the left ear → the ear-cue panel must explain it.
await page.getByRole("button", { name: "Whisper in your left ear" }).click();
await page.waitForTimeout(1500);
res.experience = await page.evaluate(() => {
  const s = window.__sonicsphere.liveRef.status;
  return { playing: s.transport.playing, trajectory: s.trajectory.active, az: s.voices[0].az, text: document.body.innerText };
});
res.experience.explainsLeft = /reaches your left ear first/.test(res.experience.text) && /the left ear is louder/.test(res.experience.text);
delete res.experience.text;
await page.screenshot({ path: `${OUT}/05_experience.png`, fullPage: true });
// Theme toggle
await page.getByRole("button", { name: /Switch to light mode/ }).click();
await page.waitForTimeout(800);
await page.screenshot({ path: `${OUT}/04_light.png` });
res.errors = errors.slice(0, 20);
console.log(JSON.stringify(res, null, 1));
await browser.close();
const n = Object.fromEntries(res.noise.map((r) => [r.az, r.ild_db]));
const checks = [
  ["left source louder in left ear (> 8 dB)", n[90] > 8],
  ["right source louder in right ear (> 8 dB)", n[270] < -8],
  ["45° between front and 90°", n[45] > n[0] && n[45] < n[90]],
  ["trajectory advances counter-clockwise", res.traj.samples.at(-1)[1] !== res.traj.samples[0][1] && res.traj.samples.at(-1)[0] > res.traj.samples[0][0]],
  ["crossfades happen while moving", res.traj.cache.crossfades > 100],
  ["method switch keeps playing", res.switch.direct === "direct" && res.switch.after === "partitioned" && res.switch.playing],
  ["corrupt upload rejected", res.badUpload?.kind === "error"],
  ["experience places the sound at the left ear", res.experience.playing && !res.experience.trajectory && Math.abs(res.experience.az - 90) < 1],
  ["ear-cue panel explains the left-ear cues", res.experience.explainsLeft],
  ["3D off: no left/right difference (|ILD| < 1 dB)", Math.abs(res.bypass.ild_db) < 1],
  ["3D on with room ambience: side still clear (> 6 dB)", res.ambience.ild_db > 6],
  ["headphone check plays in the left ear only (> 40 dB)", res.check.ratio_db > 40],
  ["audio thread load below 50 %", res.traj.timing.load < 0.5],
  ["no console errors", res.errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
process.exit(checks.every((c) => c[1]) ? 0 : 1);
