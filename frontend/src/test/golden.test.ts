/**
 * Cross-validation of the browser DSP against the Python reference implementation.
 * Golden vectors: `cd backend && python -m sonicsphere.export_golden`
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { cartToSph, quantizeDirection, rotateAboutUp, worldToHead, type Vec3 } from "../dsp/coordinates";
import { BinauralEngine, renderTrajectory } from "../dsp/engine";
import { parsePack, type FilterPack } from "../dsp/filterPack";
import { FracShifter } from "../dsp/fracshift";
import { HRIRInterpolator } from "../dsp/interpolation";
import { PeakLimiter } from "../dsp/limiter";
import { Trajectory } from "../dsp/trajectory";

const dir = join(dirname(fileURLToPath(import.meta.url)), "golden");
const goldenPath = join(dir, "golden.json");
const packPath = join(dir, "pack.ssfp");
if (!existsSync(goldenPath) || !existsSync(packPath)) {
  throw new Error("Golden vectors missing. Run: cd backend && python -m sonicsphere.export_golden");
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const G: any = JSON.parse(readFileSync(goldenPath, "utf8"));
let pack: FilterPack;

beforeAll(() => {
  const b = readFileSync(packPath);
  pack = parsePack(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
});

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};
const peak = (a: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };

describe("filter pack", () => {
  it("parses the Python pack", () => {
    expect(pack.header.format).toBe("sonicsphere-filter-pack");
    expect(pack.M).toBe(1250);
    expect(pack.L).toBe(256);
    expect(pack.T).toBe(pack.header.numTriangles);
    expect(pack.boundaryEdges.length % 2).toBe(0);
  });
});

describe("coordinates", () => {
  it("matches Python world→head, spherical angles, quantisation and rotation", () => {
    for (const c of G.coords) {
      const h = worldToHead(c.src, c.listener, c.yaw, c.pitch);
      expect(maxAbs(h, c.head)).toBeLessThan(1e-9);
      const s = cartToSph(c.head);
      expect(Math.abs(s.az - c.az)).toBeLessThan(1e-9);
      expect(Math.abs(s.el - c.el)).toBeLessThan(1e-9);
      expect(Math.abs(s.r - c.r)).toBeLessThan(1e-9);
      const q = quantizeDirection(c.az, c.el, G.quantum);
      expect(q[0]).toBeCloseTo(c.q[0], 9);
      expect(q[1]).toBeCloseTo(c.q[1], 9);
      expect(maxAbs(rotateAboutUp(c.head, 30), c.rot30)).toBeLessThan(1e-9);
    }
  });
});

describe("fractional shift", () => {
  it("matches numpy rfft/irfft implementation", () => {
    const fs = new FracShifter(G.fracshift.nfft);
    for (const c of G.fracshift.cases) {
      const out = new Float64Array(G.fracshift.nfft);
      fs.shift(G.fracshift.x, c.shift, out);
      expect(maxAbs(out, c.y)).toBeLessThan(1e-9);
    }
  });
});

describe("interpolation", () => {
  it("reproduces Python weights, statuses and filters", () => {
    const it = new HRIRInterpolator(pack, "barycentric_aligned");
    const raw = new HRIRInterpolator(pack, "barycentric_raw");
    const near = new HRIRInterpolator(pack, "nearest");
    const statuses = new Set<string>();
    for (const c of G.interp) {
      const r = it.compute(c.dir);
      statuses.add(r.w.status);
      expect(r.w.status).toBe(c.status);
      expect([...r.w.indices]).toEqual(c.indices);
      expect(maxAbs(r.w.weights, c.weights)).toBeLessThan(1e-9);
      expect(maxAbs(r.toa, c.toa)).toBeLessThan(1e-4);
      const L = pack.L;
      const scale = Math.max(peak(c.left), peak(c.right));
      expect(maxAbs(r.filter.subarray(0, L), c.left) / scale).toBeLessThan(2e-5);
      expect(maxAbs(r.filter.subarray(L), c.right) / scale).toBeLessThan(2e-5);
      expect(near.compute(c.dir).w.indices[0]).toBe(c.nearest);
      expect(maxAbs(raw.compute(c.dir).filter.subarray(0, L), c.rawLeft) / scale).toBeLessThan(2e-5);
    }
    expect(statuses).toEqual(new Set(["measured", "interpolated", "gap_fill"]));
  });

  it("caches on the quantised grid", () => {
    const it = new HRIRInterpolator(pack);
    const a = it.query([0.6, 0.7, 0.2]);
    const b = it.query([0.6, 0.7, 0.2000001]);
    expect(b).toBe(a);
    expect(it.cacheHits).toBe(1);
    expect(it.cacheMisses).toBe(1);
  });
});

describe("trajectory", () => {
  it("matches the Python spline and arc-length resampling", () => {
    const t = G.trajectory;
    for (const [name, loop] of [["closed", true], ["open", false]] as const) {
      const tr = Trajectory.fromPoints(t.points as Vec3[], t.duration, loop);
      expect(maxAbs(tr.samples, (t[name].samples as number[][]).flat())).toBeLessThan(1e-9);
      t.phases.forEach((p: number, i: number) => {
        expect(maxAbs(tr.positionAtPhase(p), t[name].at[i])).toBeLessThan(1e-9);
      });
    }
  });
});

describe("limiter", () => {
  it("matches Python sample for sample", () => {
    const x = G.limiter.x as number[][];
    const l = Float64Array.from(x.map((v) => v[0])), r = Float64Array.from(x.map((v) => v[1]));
    const lim = new PeakLimiter(G.fs, G.limiter.threshold, G.limiter.releaseMs);
    lim.process(l, r);
    const y = G.limiter.y as number[][];
    expect(maxAbs(l, y.map((v) => v[0]))).toBeLessThan(1e-9);
    expect(maxAbs(r, y.map((v) => v[1]))).toBeLessThan(1e-9);
    expect(peak(l)).toBeLessThanOrEqual(G.limiter.threshold + 1e-12);
  });
});

describe("block engine (moving source)", () => {
  for (const method of ["direct", "partitioned"] as const) {
    for (const voices of [1, 2]) {
      it(`${method}, ${voices} voice(s): output equals the Python BlockEngine`, () => {
        const E = G.engine;
        const c = E.cases.find((k: { method: string; voices: number }) => k.method === method && k.voices === voices);
        const interp = new HRIRInterpolator(pack, "barycentric_aligned");
        const eng = new BinauralEngine(interp, { blockSize: G.blockSize, method, voices, recordKeys: true });
        const traj = Trajectory.fromPoints(E.points, E.duration, E.loop);
        const sigs = voices === 1 ? [Float64Array.from(E.signal)] : [Float64Array.from(E.signal), Float64Array.from(E.signal2)];
        const [L, R] = renderTrajectory(eng, sigs, c.offsets, traj,
          { position: E.pose.position, yaw: E.pose.yaw, pitch: E.pose.pitch }, E.gain, E.distanceAttenuation);
        // Same quantised cache keys block by block.
        let keyMismatch = 0;
        eng.keys.forEach((k, i) => { if (Math.abs(k[0] - c.keys[i][0]) > 1e-9 || Math.abs(k[1] - c.keys[i][1]) > 1e-9) keyMismatch++; });
        expect(keyMismatch).toBe(0);
        expect(eng.stats.filterUpdates).toBe(c.filterUpdates);
        expect(eng.stats.crossfades).toBe(c.crossfades);
        const scale = Math.max(peak(c.left), peak(c.right));
        expect(maxAbs(L, c.left) / scale).toBeLessThan(1e-4);
        expect(maxAbs(R, c.right) / scale).toBeLessThan(1e-4);
      });
    }
  }
});
