/**
 * Interactive 3D workspace.
 *
 * Interactions
 *   position mode  click the guide sphere to place the source; drag the source orb to move it
 *   draw mode      press and drag over the sphere to record a trajectory
 *   edit mode      drag trajectory handles; click the sphere to insert a point; shift-click a handle to delete it
 *
 * The visual source follows the worklet: during trajectory playback its position is
 * derived from the phase reported by the audio thread (extrapolated between reports).
 */
import { Grid, Line, OrbitControls, PerformanceMonitor } from "@react-three/drei";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { listenerBasis, type Vec3 } from "../dsp/coordinates";
import { Trajectory } from "../dsp/trajectory";
import { HEAD, liveRef, useStore } from "../state/store";
import { Listener } from "./Listener";
import { PALETTES, type ScenePalette } from "./palette";

const H = new THREE.Vector3(...HEAD);

/** Ray–sphere intersection around the head. Falls back to the silhouette point on a miss. */
function raySphere(ray: THREE.Ray, r: number, prefer?: THREE.Vector3): THREE.Vector3 {
  const oc = ray.origin.clone().sub(H);
  const b = ray.direction.dot(oc);
  const c = oc.lengthSq() - r * r;
  const disc = b * b - c;
  if (disc < 0) {
    const p = ray.origin.clone().addScaledVector(ray.direction, -b).sub(H).setLength(r);
    return p.add(H);
  }
  const s = Math.sqrt(disc);
  const t1 = -b - s, t2 = -b + s;
  const p1 = ray.at(t1, new THREE.Vector3()), p2 = ray.at(t2, new THREE.Vector3());
  if (prefer) return p1.distanceTo(prefer) <= p2.distanceTo(prefer) ? p1 : p2;
  return t1 > 0 ? p1 : p2;
}

const toV = (p: Vec3) => new THREE.Vector3(p[0], p[1], p[2]);
const fromV = (v: THREE.Vector3): Vec3 => [v.x, v.y, v.z];

/** Text label drawn once into a texture (a GPU sprite – no per-frame DOM layout like HTML labels). */
function LabelSprite({ text, sub, color, bg, position }: { text: string; sub: string; color: string; bg: string; position: [number, number, number] }) {
  const tex = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = 256; c.height = 64;
    const g = c.getContext("2d")!;
    g.font = "600 26px Inter, sans-serif";
    const w1 = g.measureText(text).width;
    g.font = "500 22px Inter, sans-serif";
    const w2 = g.measureText(sub).width;
    const w = w1 + w2 + 46, x0 = (256 - w) / 2;
    g.fillStyle = bg;
    g.beginPath(); g.roundRect(x0, 10, w, 44, 22); g.fill();
    g.fillStyle = color;
    g.font = "600 26px Inter, sans-serif"; g.fillText(text, x0 + 18, 42);
    g.globalAlpha = 0.55; g.font = "500 22px Inter, sans-serif"; g.fillText(sub, x0 + 28 + w1, 42);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    return t;
  }, [text, sub, color, bg]);
  useEffect(() => () => tex.dispose(), [tex]);
  return (
    <sprite position={position} scale={[0.18, 0.045, 1]} renderOrder={10}>
      <spriteMaterial map={tex} transparent depthWrite={false} depthTest={false} sizeAttenuation={false} />
    </sprite>
  );
}

/** Shadows only change when the listener turns, so the shadow map is re-rendered on demand. */
function StaticShadows({ deps }: { deps: unknown[] }) {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    gl.shadowMap.autoUpdate = false;
    gl.shadowMap.needsUpdate = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gl, ...deps]);
  return null;
}

/** Wireframe guide sphere: parallels every 30°, meridians every 30°, horizon highlighted. */
function GuideSphere({ r, p, yaw }: { r: number; p: ScenePalette; yaw: number }) {
  const lines = useMemo(() => {
    const out: { pts: THREE.Vector3[]; major: boolean }[] = [];
    for (let el = -60; el <= 60; el += 30) {
      const pts: THREE.Vector3[] = [];
      for (let a = 0; a <= 360; a += 4) {
        const e = (el * Math.PI) / 180, az = (a * Math.PI) / 180;
        pts.push(new THREE.Vector3(-Math.cos(e) * Math.sin(az) * r, Math.sin(e) * r, -Math.cos(e) * Math.cos(az) * r));
      }
      out.push({ pts, major: el === 0 });
    }
    for (let a = 0; a < 180; a += 30) {
      const pts: THREE.Vector3[] = [];
      for (let t = 0; t <= 360; t += 4) {
        const th = (t * Math.PI) / 180, az = (a * Math.PI) / 180;
        pts.push(new THREE.Vector3(-Math.cos(th) * Math.sin(az) * r, Math.sin(th) * r, -Math.cos(th) * Math.cos(az) * r));
      }
      out.push({ pts, major: a === 0 });
    }
    return out;
  }, [r]);
  return (
    <group position={HEAD} rotation={[0, (yaw * Math.PI) / 180, 0]}>
      {lines.map((l, i) => (
        <Line key={i} points={l.pts} color={l.major ? p.ring : p.sphere} lineWidth={l.major ? 1.6 : 0.8}
              transparent opacity={l.major ? 0.55 : 0.16} />
      ))}
      {([["FRONT", 0], ["LEFT", 90], ["BACK", 180], ["RIGHT", 270]] as const).map(([t, az]) => {
        const a = (az * Math.PI) / 180;
        return (
          <LabelSprite key={t} text={t} sub={`${az}°`} color={p.label} bg={p.labelBg}
                       position={[-Math.sin(a) * (r + 0.28), 0, -Math.cos(a) * (r + 0.28)]} />
        );
      })}
    </group>
  );
}

/** Measured HRIR directions (head-relative, so they turn with the listener). */
function MeasurementPoints({ r, p }: { r: number; p: ScenePalette }) {
  const pack = useStore((s) => s.pack);
  const yaw = useStore((s) => s.listenerYaw), pitch = useStore((s) => s.listenerPitch);
  const ref = useRef<THREE.Group>(null);
  const geom = useMemo(() => {
    if (!pack) return null;
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(pack.M * 3);
    const [f, l, u] = listenerBasis(yaw, pitch);
    for (let i = 0; i < pack.M; i++) {
      const x = pack.vertices[3 * i], y = pack.vertices[3 * i + 1], z = pack.vertices[3 * i + 2];
      for (let c = 0; c < 3; c++) pos[3 * i + c] = (x * f[c] + y * l[c] + z * u[c]) * r * 0.995;
    }
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    return g;
  }, [pack, r, yaw, pitch]);
  const hl = useRef<THREE.InstancedMesh>(null);
  const tmp = useMemo(() => new THREE.Object3D(), []);
  useFrame(() => {
    const live = liveRef.status;
    const m = hl.current;
    if (!m || !pack) return;
    const v = live?.voices[0];
    const [f, l, u] = listenerBasis(yaw, pitch);
    for (let k = 0; k < 4; k++) {
      if (v && k < v.indices.length) {
        const i = v.indices[k];
        const x = pack.vertices[3 * i], y = pack.vertices[3 * i + 1], z = pack.vertices[3 * i + 2];
        tmp.position.set((x * f[0] + y * l[0] + z * u[0]) * r, (x * f[1] + y * l[1] + z * u[1]) * r, (x * f[2] + y * l[2] + z * u[2]) * r);
        tmp.scale.setScalar(0.4 + 0.9 * v.weights[k]);
      } else {
        tmp.position.set(0, -100, 0);
        tmp.scale.setScalar(0.001);
      }
      tmp.updateMatrix();
      m.setMatrixAt(k, tmp.matrix);
    }
    m.instanceMatrix.needsUpdate = true;
  });
  if (!geom) return null;
  return (
    <group position={HEAD} ref={ref}>
      <points geometry={geom}>
        <pointsMaterial color={p.ring} size={0.022} sizeAttenuation transparent opacity={0.5} depthWrite={false} />
      </points>
      <instancedMesh ref={hl} args={[undefined, undefined, 4]}>
        <sphereGeometry args={[0.03, 12, 12]} />
        <meshBasicMaterial color={p.source} toneMapped={false} />
      </instancedMesh>
    </group>
  );
}

function makeGlowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.25, "rgba(255,255,255,.55)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function currentVisualPosition(out: THREE.Vector3, traj: Trajectory | null): THREE.Vector3 {
  const st = useStore.getState();
  const live = liveRef.status;
  if (traj && live?.trajectory.active) {
    const dt = live.trajectory.playing ? (performance.now() - liveRef.receivedAt) / 1000 : 0;
    let ph = live.trajectory.phase + dt / Math.max(1e-3, live.trajectory.duration);
    ph = live.trajectory.loop ? ph - Math.floor(ph) : Math.min(1, ph);
    const p = traj.positionAtPhase(ph);
    return out.set(p[0], p[1], p[2]);
  }
  return out.set(st.sourcePos[0], st.sourcePos[1], st.sourcePos[2]);
}

function SourceOrb({ p, onDragStart }: { p: ScenePalette; onDragStart: (e: ThreeEvent<PointerEvent>) => void }) {
  const group = useRef<THREE.Group>(null);
  const halo = useRef<THREE.Sprite>(null);
  const ring = useRef<THREE.Mesh>(null);
  const rayLine = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color: p.ray, transparent: true, opacity: 0.75 }));
  }, [p.ray]);
  const dropLine = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color: p.ray, transparent: true, opacity: 0.25 }));
  }, [p.ray]);
  const tex = useMemo(makeGlowTexture, []);
  const samples = useStore((s) => s.traj.samples);
  const duration = useStore((s) => s.traj.duration);
  const loop = useStore((s) => s.traj.loop);
  const traj = useMemo(() => (samples ? new Trajectory(samples, duration, loop) : null), [samples, duration, loop]);
  const v = useMemo(() => new THREE.Vector3(), []);
  const [hover, setHover] = useState(false);
  useFrame((state) => {
    currentVisualPosition(v, traj);
    group.current?.position.copy(v);
    const live = liveRef.status;
    const level = live ? Math.max(live.peaks.left, live.peaks.right) : 0;
    const playing = live?.transport.playing ?? false;
    const t = state.clock.elapsedTime;
    if (halo.current) halo.current.scale.setScalar(0.42 + (playing ? 0.5 * Math.min(1, level * 1.6) : 0) + 0.04 * Math.sin(t * 3));
    if (ring.current) {
      const ph = (t * 0.9) % 1;
      ring.current.scale.setScalar(1 + ph * (playing ? 2.4 : 0.6));
      (ring.current.material as THREE.MeshBasicMaterial).opacity = (1 - ph) * (playing ? 0.6 : 0.2);
      ring.current.lookAt(state.camera.position);
    }
    {
      const a = rayLine.geometry.attributes.position as THREE.BufferAttribute;
      a.setXYZ(0, H.x, H.y, H.z); a.setXYZ(1, v.x, v.y, v.z); a.needsUpdate = true;
      rayLine.geometry.computeBoundingSphere();
      const b = dropLine.geometry.attributes.position as THREE.BufferAttribute;
      b.setXYZ(0, v.x, v.y, v.z); b.setXYZ(1, v.x, 0.002, v.z); b.needsUpdate = true;
      dropLine.geometry.computeBoundingSphere();
    }
  });
  return (
    <>
      <primitive object={rayLine} />
      <primitive object={dropLine} />
      <group ref={group}>
        <mesh onPointerDown={onDragStart} onPointerOver={() => { setHover(true); document.body.style.cursor = "grab"; }}
              onPointerOut={() => { setHover(false); document.body.style.cursor = ""; }}>
          <sphereGeometry args={[hover ? 0.11 : 0.095, 32, 24]} />
          <meshStandardMaterial color={p.source} emissive={p.sourceGlow} emissiveIntensity={1.6} toneMapped={false} />
        </mesh>
        <sprite ref={halo} scale={0.45}>
          <spriteMaterial map={tex} color={p.sourceGlow} transparent depthWrite={false} opacity={0.75} blending={THREE.AdditiveBlending} />
        </sprite>
        <mesh ref={ring}>
          <ringGeometry args={[0.13, 0.145, 48]} />
          <meshBasicMaterial color={p.source} transparent opacity={0.4} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
        <pointLight color={p.sourceGlow} intensity={1.2} distance={2.5} decay={2} />
      </group>
    </>
  );
}

function TrajectoryView({ p, editing, onHandleDown }: { p: ScenePalette; editing: boolean; onHandleDown: (i: number, e: ThreeEvent<PointerEvent>) => void }) {
  const samples = useStore((s) => s.traj.samples);
  const points = useStore((s) => s.traj.points);
  const pts = useMemo(() => {
    if (!samples) return null;
    const out: THREE.Vector3[] = [];
    for (let i = 0; i < samples.length; i += 3) out.push(new THREE.Vector3(samples[i], samples[i + 1], samples[i + 2]));
    return out;
  }, [samples]);
  if (!pts) return null;
  return (
    <group>
      <Line points={pts} color={p.path} lineWidth={2.2} transparent opacity={0.9} />
      {/* direction arrows along the path */}
      {[0.12, 0.37, 0.62, 0.87].map((f) => {
        const i = Math.floor(f * (pts.length - 2));
        const a = pts[i], b = pts[i + 1];
        const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
        return (
          <mesh key={f} position={a} quaternion={q}>
            <coneGeometry args={[0.03, 0.08, 10]} />
            <meshBasicMaterial color={p.path} />
          </mesh>
        );
      })}
      {editing && points.map((q, i) => (
        <mesh key={i} position={q} onPointerDown={(e) => onHandleDown(i, e)}
              onPointerOver={() => (document.body.style.cursor = "grab")} onPointerOut={() => (document.body.style.cursor = "")}>
          <sphereGeometry args={[0.05, 16, 12]} />
          <meshStandardMaterial color={p.handle} emissive={p.path} emissiveIntensity={0.8} />
        </mesh>
      ))}
    </group>
  );
}

/** Reduce a raw drawn stroke to evenly spaced key points (the spline control points). */
function simplifyStroke(raw: THREE.Vector3[]): Vec3[] {
  if (raw.length < 2) return raw.map(fromV);
  let total = 0;
  for (let i = 1; i < raw.length; i++) total += raw[i].distanceTo(raw[i - 1]);
  const n = Math.max(4, Math.min(40, Math.round(total / 0.3)));
  const out: Vec3[] = [fromV(raw[0])];
  const step = total / (n - 1);
  let acc = 0, next = step;
  for (let i = 1; i < raw.length; i++) {
    const d = raw[i].distanceTo(raw[i - 1]);
    while (acc + d >= next && out.length < n - 1) {
      const u = (next - acc) / d;
      out.push(fromV(raw[i - 1].clone().lerp(raw[i], u)));
      next += step;
    }
    acc += d;
  }
  out.push(fromV(raw[raw.length - 1]));
  return out;
}

function Interaction({ p, controls }: { p: ScenePalette; controls: React.MutableRefObject<{ enabled: boolean } | null> }) {
  const mode = useStore((s) => s.mode);
  const r = useStore((s) => s.sphereRadius);
  const [stroke, setStroke] = useState<THREE.Vector3[] | null>(null);
  const drag = useRef<{ kind: "source" | "handle"; index: number; radius: number; last: THREE.Vector3 } | null>(null);
  const drawing = useRef<THREE.Vector3[] | null>(null);
  const lastSend = useRef(0);

  const setControls = (on: boolean) => { if (controls.current) controls.current.enabled = on; };

  const startSourceDrag = (e: ThreeEvent<PointerEvent>) => {
    if (mode !== "position") return;
    e.stopPropagation();
    const st = useStore.getState();
    if (st.trajPlaying || liveRef.status?.trajectory.active) st.stopTrajectory();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const cur = toV(st.sourcePos);
    drag.current = { kind: "source", index: -1, radius: Math.max(0.3, cur.distanceTo(H)), last: cur };
    setControls(false);
    document.body.style.cursor = "grabbing";
  };

  const startHandleDrag = (i: number, e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const st = useStore.getState();
    if (e.shiftKey) {
      const pts = st.traj.points.filter((_, k) => k !== i);
      st.setTrajectory(pts.length >= 2 ? pts : []);
      return;
    }
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const cur = toV(st.traj.points[i]);
    drag.current = { kind: "handle", index: i, radius: Math.max(0.3, cur.distanceTo(H)), last: cur };
    setControls(false);
  };

  const onMove = (e: ThreeEvent<PointerEvent>) => {
    const d = drag.current;
    if (d) {
      const q = raySphere(e.ray, d.radius, d.last);
      d.last = q;
      const st = useStore.getState();
      if (d.kind === "source") st.setSourcePos(fromV(q));
      else {
        const now = performance.now();
        const pts = st.traj.points.slice();
        pts[d.index] = fromV(q);
        if (now - lastSend.current > 40) { lastSend.current = now; st.setTrajectory(pts); }
        else useStore.setState({ traj: { ...st.traj, points: pts } });
      }
      return;
    }
    if (drawing.current) {
      const q = raySphere(e.ray, r);
      const last = drawing.current[drawing.current.length - 1];
      if (!last || last.distanceTo(q) > 0.03) {
        drawing.current.push(q);
        setStroke([...drawing.current]);
      }
    }
  };

  const onUp = (e: ThreeEvent<PointerEvent>) => {
    (e.target as Element).releasePointerCapture?.(e.pointerId);
    const st = useStore.getState();
    if (drag.current?.kind === "handle") st.setTrajectory(st.traj.points);
    drag.current = null;
    document.body.style.cursor = "";
    if (drawing.current) {
      const raw = drawing.current;
      drawing.current = null;
      setStroke(null);
      if (raw.length >= 3) {
        st.setTrajectory(simplifyStroke(raw), "Custom drawing");
        st.setMode("edit");
        st.notify("info", "Trajectory recorded. Drag the handles to edit it, then press Play path.");
      }
    }
    setControls(true);
  };

  const onSphereDown = (e: ThreeEvent<PointerEvent>) => {
    if (mode === "draw") {
      e.stopPropagation();
      (e.target as Element).setPointerCapture?.(e.pointerId);
      drawing.current = [e.point.clone()];
      setStroke([e.point.clone()]);
      setControls(false);
    }
  };

  const onSphereClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 5) return; // that was an orbit drag, not a click
    const st = useStore.getState();
    if (mode === "position") {
      e.stopPropagation();
      if (st.trajPlaying || liveRef.status?.trajectory.active) st.stopTrajectory();
      st.setSourcePos(fromV(e.point));
    } else if (mode === "edit" && st.traj.points.length >= 2) {
      e.stopPropagation();
      const pts = st.traj.points;
      let best = 0, bd = Infinity;
      pts.forEach((q, i) => { const d2 = e.point.distanceTo(toV(q)); if (d2 < bd) { bd = d2; best = i; } });
      const copy = pts.slice();
      copy.splice(best + 1, 0, fromV(e.point));
      st.setTrajectory(copy);
    }
  };

  return (
    <group onPointerMove={onMove} onPointerUp={onUp}>
      <mesh position={HEAD} onPointerDown={onSphereDown} onClick={onSphereClick}
            onPointerOver={() => { if (mode !== "position") document.body.style.cursor = "crosshair"; }}
            onPointerOut={() => { if (!drag.current) document.body.style.cursor = ""; }}>
        <sphereGeometry args={[r, 48, 32]} />
        <meshBasicMaterial transparent opacity={mode === "draw" ? 0.05 : 0.0} color={p.sphere} depthWrite={false} side={THREE.FrontSide} />
      </mesh>
      <SourceOrb p={p} onDragStart={startSourceDrag} />
      <TrajectoryView p={p} editing={mode === "edit"} onHandleDown={startHandleDrag} />
      {stroke && stroke.length > 1 && <Line points={stroke} color={p.pathDraw} lineWidth={3} />}
    </group>
  );
}

function SceneContent({ controls }: { controls: React.MutableRefObject<{ enabled: boolean } | null> }) {
  const theme = useStore((s) => s.theme);
  const p = PALETTES[theme];
  const yaw = useStore((s) => s.listenerYaw), pitch = useStore((s) => s.listenerPitch);
  const r = useStore((s) => s.sphereRadius);
  const showGrid = useStore((s) => s.pack !== null);
  return (
    <>
      <color attach="background" args={[p.bg]} />
      <fog attach="fog" args={[p.fog, 7, 18]} />
      <ambientLight intensity={p.ambient} />
      <hemisphereLight args={["#bcd3ff", "#1a1020", 0.35]} />
      <directionalLight position={[3.5, 6, 2.5]} intensity={p.key} castShadow shadow-mapSize={[1024, 1024]}
                        shadow-camera-left={-3} shadow-camera-right={3} shadow-camera-top={3} shadow-camera-bottom={-3} />
      <pointLight position={[-3, 2.5, -2]} intensity={0.6} color="#7dd3fc" />
      <directionalLight position={[0.5, 2.2, -3]} intensity={0.9} color="#ffffff" />
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[0, -0.001, 0]}>
        <circleGeometry args={[6, 64]} />
        <meshStandardMaterial color={p.floor} roughness={0.95} />
      </mesh>
      <Grid position={[0, 0.001, 0]} args={[20, 20]} cellSize={0.25} cellThickness={0.6} cellColor={p.grid}
            sectionSize={1} sectionThickness={1.1} sectionColor={p.gridSection} fadeDistance={14} fadeStrength={1.4} infiniteGrid />
      {/* forward arrow on the floor */}
      <group rotation={[0, (yaw * Math.PI) / 180, 0]}>
        <mesh position={[0, 0.005, -0.95]} rotation={[-Math.PI / 2, 0, Math.PI]}>
          <coneGeometry args={[0.09, 0.22, 3]} />
          <meshBasicMaterial color={p.front} transparent opacity={0.7} />
        </mesh>
      </group>
      <Listener p={p} yaw={yaw} pitch={pitch} />
      <StaticShadows deps={[yaw, pitch, theme]} />
      <GuideSphere r={r} p={p} yaw={yaw} />
      {showGrid && <MeasurementPoints r={r} p={p} />}
      <Interaction p={p} controls={controls} />
      <OrbitControls ref={controls as never} target={HEAD} enableDamping dampingFactor={0.08} minDistance={1.4} maxDistance={11}
                     maxPolarAngle={Math.PI * 0.94} makeDefault />
    </>
  );
}

export function Scene() {
  const controls = useRef<{ enabled: boolean } | null>(null);
  const [lost, setLost] = useState(false);
  // Adaptive resolution: start at up to 1.5x, drop towards 1x if the frame rate falls (laptops).
  const [dpr, setDpr] = useState(() => Math.min(1.5, window.devicePixelRatio || 1));
  useEffect(() => () => { document.body.style.cursor = ""; }, []);
  // Pause 3D rendering while the view is scrolled off screen or the tab is hidden.
  const box = useRef<HTMLDivElement>(null);
  const [onScreen, setOnScreen] = useState(true);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setOnScreen(e.isIntersecting), { threshold: 0.02 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={box} className="relative h-full w-full">
      <Canvas shadows dpr={dpr} frameloop={onScreen ? "always" : "never"} camera={{ position: [3.4, 2.3, 1.9], fov: 45, near: 0.05, far: 60 }}
              gl={{ antialias: true, powerPreference: "high-performance" }}
              onCreated={({ gl }) => {
                gl.domElement.addEventListener("webglcontextlost", (ev) => { ev.preventDefault(); setLost(true); });
                gl.domElement.addEventListener("webglcontextrestored", () => setLost(false));
              }}>
        <PerformanceMonitor onDecline={() => setDpr((d) => Math.max(1, d - 0.25))} onIncline={() => setDpr((d) => Math.min(1.5, window.devicePixelRatio || 1, d + 0.25))} flipflops={3} onFallback={() => setDpr(1)} />
        <SceneContent controls={controls} />
      </Canvas>
      {lost && (
        <div className="absolute inset-0 grid place-items-center bg-black/60 text-sm text-white">
          The 3D view lost its graphics context. Audio keeps running; reload the page to restore the view.
        </div>
      )}
    </div>
  );
}
