/** Self-contained unit tests of the browser DSP (no golden data needed). */
import { describe, expect, it } from "vitest";
import { ComplexFFT, RealFFT } from "../dsp/fft";
import { DirectConvolver, PartitionedConvolver } from "../dsp/convolver";
import { listenerBasis, quantizeDirection, worldToHead, cartToSph } from "../dsp/coordinates";
import { analyseStereo, prepareSource } from "../dsp/sourcePrep";
import { Trajectory } from "../dsp/trajectory";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (1664525 * s + 1013904223) >>> 0; return s / 2 ** 32 - 0.5; };
}

function naiveConv(x: ArrayLike<number>, h: ArrayLike<number>): Float64Array {
  const y = new Float64Array(x.length + h.length - 1);
  for (let n = 0; n < x.length; n++) for (let k = 0; k < h.length; k++) y[n + k] += x[n] * h[k];
  return y;
}

describe("FFT", () => {
  it("complex FFT matches the DFT definition and inverts", () => {
    const n = 64, r = rng(1);
    const re = new Float64Array(n).map(r), im = new Float64Array(n).map(r);
    const re0 = re.slice(), im0 = im.slice();
    new ComplexFFT(n).transform(re, im);
    for (const k of [0, 1, 7, 31, 63]) {
      let sr = 0, si = 0;
      for (let t = 0; t < n; t++) {
        const a = (-2 * Math.PI * k * t) / n;
        sr += re0[t] * Math.cos(a) - im0[t] * Math.sin(a);
        si += re0[t] * Math.sin(a) + im0[t] * Math.cos(a);
      }
      expect(Math.abs(re[k] - sr)).toBeLessThan(1e-9);
      expect(Math.abs(im[k] - si)).toBeLessThan(1e-9);
    }
    new ComplexFFT(n).transform(re, im, true);
    for (let t = 0; t < n; t++) expect(Math.abs(re[t] / n - re0[t])).toBeLessThan(1e-12);
  });

  it("real FFT matches the DFT and round-trips", () => {
    for (const n of [2, 4, 16, 256, 512]) {
      const r = rng(n), x = new Float64Array(n).map(r);
      const fft = new RealFFT(n);
      const Xr = new Float64Array(n / 2 + 1), Xi = new Float64Array(n / 2 + 1);
      fft.forward(x, Xr, Xi);
      for (let k = 0; k <= n / 2; k++) {
        let sr = 0, si = 0;
        for (let t = 0; t < n; t++) { sr += x[t] * Math.cos((2 * Math.PI * k * t) / n); si -= x[t] * Math.sin((2 * Math.PI * k * t) / n); }
        expect(Math.abs(Xr[k] - sr)).toBeLessThan(1e-9);
        expect(Math.abs(Xi[k] - si)).toBeLessThan(1e-9);
      }
      const y = new Float64Array(n);
      fft.inverse(Xr, Xi, y);
      for (let t = 0; t < n; t++) expect(Math.abs(y[t] - x[t])).toBeLessThan(1e-12);
    }
  });

  it("rejects non power-of-two sizes", () => {
    expect(() => new RealFFT(100)).toThrow();
  });
});

describe("streaming convolvers", () => {
  it("direct and partitioned equal linear convolution, block by block", () => {
    const B = 128, L = 256, r = rng(3);
    const h = new Float32Array(2 * L).map(r);
    const x = new Float64Array(B * 12).map(r);
    const refL = naiveConv(x, h.subarray(0, L)), refR = naiveConv(x, h.subarray(L));
    const d = new DirectConvolver(L, B), p = new PartitionedConvolver(L, B);
    const H = p.partition(h);
    const o = new Float64Array(B);
    for (let b = 0; b < 12; b++) {
      const blk = x.subarray(b * B, (b + 1) * B);
      d.push(blk); p.push(blk);
      d.apply(h, 0, o);
      for (let i = 0; i < B; i++) expect(Math.abs(o[i] - refL[b * B + i])).toBeLessThan(1e-9);
      d.apply(h, L, o);
      for (let i = 0; i < B; i++) expect(Math.abs(o[i] - refR[b * B + i])).toBeLessThan(1e-9);
      p.apply(H, 0, o);
      for (let i = 0; i < B; i++) expect(Math.abs(o[i] - refL[b * B + i])).toBeLessThan(1e-6);
      p.apply(H, 1, o);
      for (let i = 0; i < B; i++) expect(Math.abs(o[i] - refR[b * B + i])).toBeLessThan(1e-6);
    }
  });

  it("impulse in → filter out (left/right kept apart)", () => {
    const B = 128, L = 256;
    const h = new Float32Array(2 * L);
    h[3] = 1; h[L + 40] = -0.5;
    const d = new DirectConvolver(L, B);
    const x = new Float64Array(B); x[0] = 1;
    d.push(x);
    const o = new Float64Array(B);
    d.apply(h, 0, o); expect(o[3]).toBeCloseTo(1, 12); expect(o[40]).toBe(0);
    d.apply(h, L, o); expect(o[40]).toBeCloseTo(-0.5, 12); expect(o[3]).toBe(0);
  });
});

describe("coordinates", () => {
  it("listener basis is orthonormal and right handed", () => {
    for (const yaw of [0, 33, -150]) for (const pitch of [0, 25]) {
      const [f, l, u] = listenerBasis(yaw, pitch);
      const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
      expect(dot(f, l)).toBeCloseTo(0, 12); expect(dot(f, u)).toBeCloseTo(0, 12); expect(dot(u, u)).toBeCloseTo(1, 12);
    }
  });
  it("world −X is the left of an unrotated listener; turning changes the direction", () => {
    expect(cartToSph(worldToHead([-1, 1.2, 0], [0, 1.2, 0], 0)).az).toBeCloseTo(90, 9);
    expect(cartToSph(worldToHead([-1, 1.2, 0], [0, 1.2, 0], -90)).az).toBeCloseTo(180, 9);
    expect(cartToSph(worldToHead([0, 3, 0], [0, 1, 0], 0)).el).toBeCloseTo(90, 9);
  });
  it("quantises with wrap-around", () => {
    expect(quantizeDirection(359.97, 10, 0.1)).toEqual([0, 10]);
    expect(quantizeDirection(10, 89.99, 0.1)).toEqual([0, 90]);
  });
});

describe("trajectory", () => {
  it("loops, clamps and moves at uniform speed", () => {
    const tr = Trajectory.fromPoints([[1, 0, 0], [0, 0, -1], [-1, 0, 0], [0, 0, 1]], 2, true);
    const a = tr.positionAt(0.3), b = tr.positionAt(2.3);
    for (let i = 0; i < 3; i++) expect(a[i]).toBeCloseTo(b[i], 12);
    const seg: number[] = [];
    for (let i = 1; i < tr.count; i++) seg.push(Math.hypot(...[0, 1, 2].map((c) => tr.samples[3 * i + c] - tr.samples[3 * i - 3 + c])));
    expect(Math.max(...seg) / Math.min(...seg)).toBeLessThan(1.05);
    const once = Trajectory.fromPoints([[0, 0, 0], [2, 0, 0]], 1, false);
    expect(once.positionAt(5)[0]).toBeCloseTo(2, 12);
    expect(once.positionAt(0.5)[0]).toBeCloseTo(1, 9);
  });
});

describe("source preparation", () => {
  it("applies the stereo policies and reports what the downmix removes", () => {
    const r = rng(9), n = 4000;
    const l = new Float32Array(n).map(r), rr = new Float32Array(n).map(r);
    const mid = prepareSource([l, rr], "mid", 30);
    expect(mid.signals.length).toBe(1);
    expect(mid.analysis.sideEnergyFraction!).toBeGreaterThan(0.4);
    const pair = prepareSource([l, rr], "stereo_pair", 40);
    expect(pair.offsetsDeg).toEqual([20, -20]);
    expect(prepareSource([l], "stereo_pair", 40).policy).toBe("mono");
    const anti = analyseStereo([l, l.map((v) => -v)]);
    expect(anti.warning).toBeTruthy();
  });
  it("normalises to the RMS target with a peak ceiling; leaves silence alone", () => {
    const r = rng(2), x = new Float32Array(44100).map((_) => 0.01 * r());
    const p = prepareSource([x], "mid", 30, -20, -1);
    expect(20 * Math.log10(Math.sqrt(p.signals[0].reduce((a, v) => a + v * v, 0) / x.length))).toBeCloseTo(-20, 2);
    const click = new Float32Array(1000); click[10] = 0.2;
    expect(Math.max(...prepareSource([click], "mid", 30).signals[0])).toBeCloseTo(Math.pow(10, -1 / 20), 6);
    const silent = prepareSource([new Float32Array(300)], "mid", 30);
    expect(silent.silent).toBe(true);
    expect(silent.normalisationGain).toBe(1);
  });
});

describe("utf-8 fallback (AudioWorkletGlobalScope has no TextDecoder)", () => {
  it("matches TextDecoder", async () => {
    const { decodeUtf8 } = await import("../dsp/filterPack");
    const txt = '{"a":"ASCII","b":"× – ° µ 😀"}';
    const bytes = new TextEncoder().encode(txt);
    const saved = globalThis.TextDecoder;
    // @ts-expect-error simulate the worklet scope
    delete globalThis.TextDecoder;
    try { expect(decodeUtf8(bytes)).toBe(txt); } finally { globalThis.TextDecoder = saved; }
  });
});
