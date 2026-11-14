/**
 * Parser for the binary filter pack produced by backend/sonicsphere/filter_pack.py.
 *
 *   0  "SSFP" | 4 u32 version | 8 u32 header bytes H | 12 u32 reserved
 *   16 JSON header (H bytes) | data region (16-byte aligned arrays, offsets in the header)
 */

export interface PackArrayMeta { dtype: "float32" | "float64" | "uint32" | "uint8"; shape: number[]; offset: number; byteLength: number }

export interface PackHeader {
  format: string;
  version: number;
  dataset: string;
  datasetName: string;
  subject: string;
  sha256: string;
  fs: number;
  sourceFs: number;
  sourceLength: number;
  filterLength: number;
  margin: number;
  blockSize: number;
  gain: number;
  numDirections: number;
  numTriangles: number;
  numGapTriangles: number;
  gapEdgeThresholdDeg: number;
  measuredSnapDeg: number;
  directionQuantumDeg: number;
  interpolation: string;
  limiterThreshold: number;
  limiterReleaseMs: number;
  onsetThresholdDb: number;
  onsetUpsample: number;
  truncatedEnergyDb: number;
  fracShiftFft: number;
  grid: Record<string, unknown>;
  notes: string[];
  arrays: Record<string, PackArrayMeta>;
}

export interface FilterPack {
  header: PackHeader;
  M: number;            // directions
  T: number;            // triangles
  L: number;            // filter length
  measured: Float32Array;   // (M, 2, L)
  aligned: Float32Array;    // (M, 2, L)
  toa: Float32Array;        // (M, 2)
  vertices: Float64Array;   // (M, 3)
  azel: Float32Array;       // (M, 2)
  triangles: Uint32Array;   // (T, 3)
  triInverse: Float64Array; // (T, 9) row-major
  triMaxEdge: Float32Array; // (T,)
  triGap: Uint8Array;       // (T,)
  boundaryEdges: Uint32Array; // (E, 2)
  byteLength: number;
}

/** UTF-8 decode that also works in AudioWorkletGlobalScope, which has no TextDecoder. */
export function decodeUtf8(b: Uint8Array): string {
  if (typeof TextDecoder !== "undefined") return new TextDecoder().decode(b);
  let out = "";
  for (let i = 0; i < b.length;) {
    const c = b[i++];
    let cp: number;
    if (c < 0x80) cp = c;
    else if (c < 0xe0) cp = ((c & 0x1f) << 6) | (b[i++] & 0x3f);
    else if (c < 0xf0) cp = ((c & 0x0f) << 12) | ((b[i++] & 0x3f) << 6) | (b[i++] & 0x3f);
    else cp = ((c & 0x07) << 18) | ((b[i++] & 0x3f) << 12) | ((b[i++] & 0x3f) << 6) | (b[i++] & 0x3f);
    out += String.fromCodePoint(cp);
  }
  return out;
}

const CTORS = { float32: Float32Array, float64: Float64Array, uint32: Uint32Array, uint8: Uint8Array } as const;

export function parsePack(buf: ArrayBuffer): FilterPack {
  const u8 = new Uint8Array(buf);
  if (buf.byteLength < 16 || String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "SSFP") {
    throw new Error("Not a SonicSphere filter pack (bad magic)");
  }
  const dv = new DataView(buf);
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`Unsupported filter pack version ${version}`);
  const hlen = dv.getUint32(8, true);
  const header = JSON.parse(decodeUtf8(u8.subarray(16, 16 + hlen))) as PackHeader;
  const base = 16 + hlen;
  const get = <K extends keyof typeof CTORS>(name: string, dtype: K): InstanceType<(typeof CTORS)[K]> => {
    const m = header.arrays[name];
    if (!m) throw new Error(`Filter pack is missing array '${name}'`);
    if (m.dtype !== dtype) throw new Error(`Array '${name}' has dtype ${m.dtype}, expected ${dtype}`);
    const Ctor = CTORS[dtype];
    const count = m.byteLength / Ctor.BYTES_PER_ELEMENT;
    if ((base + m.offset) % Ctor.BYTES_PER_ELEMENT === 0) {
      return new Ctor(buf, base + m.offset, count) as InstanceType<(typeof CTORS)[K]>;
    }
    return new Ctor(buf.slice(base + m.offset, base + m.offset + m.byteLength)) as InstanceType<(typeof CTORS)[K]>;
  };
  const pack: FilterPack = {
    header,
    M: header.numDirections,
    T: header.numTriangles,
    L: header.filterLength,
    measured: get("measured", "float32"),
    aligned: get("aligned", "float32"),
    toa: get("toa", "float32"),
    vertices: get("vertices", "float64"),
    azel: get("azel", "float32"),
    triangles: get("triangles", "uint32"),
    triInverse: get("triInverse", "float64"),
    triMaxEdge: get("triMaxEdge", "float32"),
    triGap: get("triGap", "uint8"),
    boundaryEdges: get("boundaryEdges", "uint32"),
    byteLength: buf.byteLength,
  };
  if (pack.measured.length !== pack.M * 2 * pack.L) throw new Error("Filter pack: inconsistent HRIR array size");
  if (pack.triangles.length !== pack.T * 3) throw new Error("Filter pack: inconsistent triangle array size");
  if (pack.L % header.blockSize !== 0) throw new Error("Filter pack: filter length is not a multiple of the block size");
  return pack;
}
