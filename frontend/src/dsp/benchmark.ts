/**
 * In-browser micro-benchmark of the real-time kernels (same code as the AudioWorklet).
 * Results are labelled as benchmark measurements, never as live runtime values.
 */

import { sphToCart } from "./coordinates";
import { DirectConvolver, PartitionedConvolver, reverseTaps, type ConvMethod } from "./convolver";
import { BinauralEngine } from "./engine";
import type { FilterPack } from "./filterPack";
import { HRIRInterpolator, type InterpMethod } from "./interpolation";

export interface KernelResult { method: ConvMethod; usPerBlock: number; p95: number }
export interface EngineResult { method: ConvMethod; scenario: "static" | "moving"; voices: number; usPerBlock: number; p95: number; max: number }
export interface InterpResult { method: InterpMethod; usPerQuery: number }

export interface BenchmarkReport {
  when: string;
  fs: number;
  blockSize: number;
  filterLength: number;
  deadlineUs: number;
  timerResolutionUs: number;
  kernels: KernelResult[];
  engine: EngineResult[];
  interpolation: InterpResult[];
  recommended: ConvMethod;
  reason: string;
}

const now = () => performance.now();

function stats(samples: number[]) {
  const s = [...samples].sort((a, b) => a - b);
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  return { mean, p95: s[Math.floor(0.95 * (s.length - 1))], max: s[s.length - 1] };
}

function timerResolution(): number {
  let min = Infinity;
  for (let i = 0; i < 50; i++) {
    const a = now();
    let b = now();
    while (b === a) b = now();
    min = Math.min(min, b - a);
  }
  return min * 1000;
}

/** Measure `fn` over `n` runs grouped in batches (robust to coarse timers). */
function timePerRun(fn: () => void, n: number, batch: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i += batch) {
    const t0 = now();
    for (let k = 0; k < batch; k++) fn();
    out.push(((now() - t0) * 1000) / batch);
  }
  return out;
}

export function runBenchmark(pack: FilterPack, blocks = 600): BenchmarkReport {
  const B = pack.header.blockSize, L = pack.L, fs = pack.header.fs;
  const deadline = (B / fs) * 1e6;
  const x = new Float64Array(B).map(() => Math.random() - 0.5);
  const filt = pack.measured.subarray(0, 2 * L);
  const out = new Float64Array(B);

  // Kernel: two ears, two filters (crossfade worst case).
  const kernels: KernelResult[] = [];
  {
    const d = new DirectConvolver(L, B);
    const rv = reverseTaps(filt, L);
    const fn = () => { d.push(x); d.applyReversed(rv, 0, out); d.applyReversed(rv, L, out); d.applyReversed(rv, 0, out); d.applyReversed(rv, L, out); };
    timePerRun(fn, 200, 20);
    const s = stats(timePerRun(fn, blocks, 20));
    kernels.push({ method: "direct", usPerBlock: s.mean, p95: s.p95 });
  }
  {
    const p = new PartitionedConvolver(L, B);
    const H = p.partition(filt);
    const fn = () => { p.push(x); p.apply(H, 0, out); p.apply(H, 1, out); p.apply(H, 0, out); p.apply(H, 1, out); };
    timePerRun(fn, 200, 20);
    const s = stats(timePerRun(fn, blocks, 20));
    kernels.push({ method: "partitioned", usPerBlock: s.mean, p95: s.p95 });
  }

  // Full engine: static source vs. a source that moves every block (new filter + crossfade).
  const engine: EngineResult[] = [];
  const outL = new Float32Array(B), outR = new Float32Array(B);
  for (const method of ["direct", "partitioned"] as ConvMethod[]) {
    for (const voices of [1, 2]) {
      for (const scenario of ["static", "moving"] as const) {
        const interp = new HRIRInterpolator(pack, "barycentric_aligned", 4096);
        const eng = new BinauralEngine(interp, { blockSize: B, method, voices });
        const ins = Array.from({ length: voices }, () => x);
        let k = 0;
        const fn = () => {
          const az = scenario === "moving" ? (k * 0.37) % 360 : 40;
          const el = scenario === "moving" ? 20 * Math.sin(k * 0.01) : 10;
          const d = sphToCart(az, el);
          k++;
          eng.processBlock(ins, Array.from({ length: voices }, () => d), Array.from({ length: voices }, () => 0.5), outL, outR);
        };
        timePerRun(fn, 100, 10);
        interp.clearCache();
        const s = stats(timePerRun(fn, blocks, 10));
        engine.push({ method, scenario, voices, usPerBlock: s.mean, p95: s.p95, max: s.max });
      }
    }
  }

  // Interpolation cost per (uncached) filter computation.
  const interpolation: InterpResult[] = [];
  for (const m of ["nearest", "barycentric_raw", "barycentric_aligned"] as InterpMethod[]) {
    const it = new HRIRInterpolator(pack, m);
    let k = 0;
    const fn = () => { it.compute(sphToCart((k * 7.3) % 360, ((k * 3.1) % 140) - 50)); k++; };
    timePerRun(fn, 50, 10);
    interpolation.push({ method: m, usPerQuery: stats(timePerRun(fn, 300, 10)).mean });
  }

  const mv = (m: ConvMethod) => engine.find((e) => e.method === m && e.scenario === "moving" && e.voices === 1)!;
  const recommended: ConvMethod = mv("partitioned").usPerBlock < mv("direct").usPerBlock ? "partitioned" : "direct";
  const other = recommended === "direct" ? "partitioned" : "direct";
  return {
    when: new Date().toISOString(),
    fs, blockSize: B, filterLength: L, deadlineUs: deadline,
    timerResolutionUs: timerResolution(),
    kernels, engine, interpolation, recommended,
    reason: `moving-source engine: ${recommended} ${mv(recommended).usPerBlock.toFixed(1)} µs/block vs ${other} ` +
            `${mv(other).usPerBlock.toFixed(1)} µs/block (deadline ${deadline.toFixed(0)} µs)`,
  };
}
