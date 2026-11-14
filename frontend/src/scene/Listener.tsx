/**
 * Stylised seated listener wearing headphones, built from primitives (no external model).
 * The figure faces −Z (the listener's forward direction at yaw 0). The head centre is at
 * HEAD = (0, 1.15, 0): the point the HRIRs are referenced to.
 */
import { useMemo } from "react";
import * as THREE from "three";
import { HEAD } from "../state/store";
import type { ScenePalette } from "./palette";

type V = [number, number, number];

function Limb({ a, b, r, color, rough = 0.7 }: { a: V; b: V; r: number; color: string; rough?: number }) {
  const { pos, quat, len } = useMemo(() => {
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
    const d = vb.clone().sub(va);
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.clone().normalize());
    return { pos: va.clone().add(vb).multiplyScalar(0.5), quat: q, len: d.length() };
  }, [a, b]);
  return (
    <mesh position={pos} quaternion={quat} castShadow>
      <capsuleGeometry args={[r, Math.max(0.001, len), 8, 16]} />
      <meshStandardMaterial color={color} roughness={rough} />
    </mesh>
  );
}

function Chair({ p }: { p: ScenePalette }) {
  const legs = [0, 1, 2, 3, 4].map((i) => (i / 5) * Math.PI * 2 + 0.3);
  return (
    <group>
      {/* seat */}
      <mesh position={[0, 0.47, 0.02]} castShadow receiveShadow>
        <boxGeometry args={[0.52, 0.09, 0.52]} />
        <meshStandardMaterial color={p.chair} roughness={0.55} />
      </mesh>
      {/* backrest (behind the listener = +Z) */}
      <group position={[0, 0.8, 0.29]} rotation={[-0.12, 0, 0]}>
        <mesh castShadow>
          <boxGeometry args={[0.5, 0.52, 0.08]} />
          <meshStandardMaterial color={p.chair} roughness={0.55} />
        </mesh>
        <mesh position={[0, 0, -0.045]}>
          <boxGeometry args={[0.06, 0.42, 0.01]} />
          <meshStandardMaterial color={p.chairAccent} emissive={p.chairAccent} emissiveIntensity={0.6} />
        </mesh>
      </group>
      {/* armrests */}
      {[-1, 1].map((s) => (
        <group key={s}>
          <mesh position={[s * 0.3, 0.68, -0.02]} castShadow>
            <boxGeometry args={[0.06, 0.04, 0.36]} />
            <meshStandardMaterial color={p.chair} roughness={0.5} />
          </mesh>
          <mesh position={[s * 0.3, 0.58, 0.08]}>
            <boxGeometry args={[0.035, 0.18, 0.035]} />
            <meshStandardMaterial color={p.metal} metalness={0.7} roughness={0.35} />
          </mesh>
        </group>
      ))}
      {/* gas lift and star base */}
      <mesh position={[0, 0.27, 0.02]}>
        <cylinderGeometry args={[0.03, 0.035, 0.36, 16]} />
        <meshStandardMaterial color={p.metal} metalness={0.8} roughness={0.25} />
      </mesh>
      {legs.map((a, i) => (
        <group key={i} position={[0, 0.08, 0.02]} rotation={[0, a, 0]}>
          <mesh position={[0.17, 0, 0]} castShadow>
            <boxGeometry args={[0.34, 0.035, 0.05]} />
            <meshStandardMaterial color={p.chair} roughness={0.5} />
          </mesh>
          <mesh position={[0.33, -0.045, 0]}>
            <sphereGeometry args={[0.03, 12, 12]} />
            <meshStandardMaterial color="#0b0f19" roughness={0.4} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

function Headphones({ p }: { p: ScenePalette }) {
  return (
    <group>
      {/* headband: half torus in the coronal plane, over the top of the head */}
      <mesh rotation={[0, 0, 0]}>
        <torusGeometry args={[0.128, 0.013, 12, 48, Math.PI]} />
        <meshStandardMaterial color={p.band} metalness={0.4} roughness={0.35} />
      </mesh>
      {([[-1, p.cupLeft, "L"], [1, p.cupRight, "R"]] as const).map(([s, c]) => (
        <group key={s} position={[s * 0.122, -0.005, 0]} rotation={[0, 0, Math.PI / 2]}>
          <mesh castShadow>
            <cylinderGeometry args={[0.055, 0.058, 0.045, 32]} />
            <meshStandardMaterial color={p.band} metalness={0.5} roughness={0.3} />
          </mesh>
          {/* outer ring glows: cyan = left ear, pink = right ear */}
          {/* group is rotated +90° about Z, so local +Y points to world −X: outward = −s */}
          <mesh position={[0, -s * 0.024, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.04, 0.006, 8, 32]} />
            <meshStandardMaterial color={c} emissive={c} emissiveIntensity={1.4} toneMapped={false} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

export function Listener({ p, yaw, pitch }: { p: ScenePalette; yaw: number; pitch: number }) {
  const deg = Math.PI / 180;
  const hy = HEAD[1];
  return (
    <group rotation={[0, yaw * deg, 0]}>
      <Chair p={p} />
      {/* torso */}
      <Limb a={[0, 0.6, 0.06]} b={[0, 0.93, 0.06]} r={0.16} color={p.hoodie} />
      <mesh position={[0, 0.98, 0.05]} castShadow>
        <sphereGeometry args={[0.175, 24, 16, 0, Math.PI * 2, 0, Math.PI / 2]} />
        <meshStandardMaterial color={p.hoodie} roughness={0.8} />
      </mesh>
      {/* legs: thighs forward (−Z), shins down */}
      {[-1, 1].map((s) => (
        <group key={s}>
          <Limb a={[s * 0.1, 0.56, 0.0]} b={[s * 0.11, 0.56, -0.38]} r={0.075} color={p.pants} />
          <Limb a={[s * 0.11, 0.55, -0.4]} b={[s * 0.12, 0.12, -0.44]} r={0.06} color={p.pants} />
          <mesh position={[s * 0.12, 0.04, -0.5]} castShadow>
            <boxGeometry args={[0.1, 0.07, 0.24]} />
            <meshStandardMaterial color={p.shoe} roughness={0.6} />
          </mesh>
          {/* arms resting on the armrests */}
          <Limb a={[s * 0.22, 0.98, 0.06]} b={[s * 0.27, 0.74, 0.0]} r={0.05} color={p.hoodie} />
          <Limb a={[s * 0.27, 0.73, 0.0]} b={[s * 0.25, 0.72, -0.27]} r={0.043} color={p.hoodie} />
          <mesh position={[s * 0.24, 0.72, -0.32]}>
            <sphereGeometry args={[0.045, 16, 12]} />
            <meshStandardMaterial color={p.skin} roughness={0.6} />
          </mesh>
        </group>
      ))}
      {/* neck */}
      <Limb a={[0, 1.0, 0.03]} b={[0, 1.07, 0.01]} r={0.045} color={p.skin} />
      {/* head (pivot at the head centre for pitch) */}
      <group position={[0, hy, 0]} rotation={[pitch * deg, 0, 0]}>
        <mesh castShadow>
          <sphereGeometry args={[0.105, 32, 24]} />
          <meshStandardMaterial color={p.skin} roughness={0.55} />
        </mesh>
        {/* hair cap */}
        <mesh position={[0, 0.02, 0.012]} rotation={[-0.35, 0, 0]}>
          <sphereGeometry args={[0.109, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2.3]} />
          <meshStandardMaterial color="#2a1d17" roughness={0.9} />
        </mesh>
        {/* nose and visor mark the forward direction (−Z) */}
        <mesh position={[0, -0.01, -0.105]} rotation={[-Math.PI / 2, 0, 0]}>
          <coneGeometry args={[0.018, 0.04, 12]} />
          <meshStandardMaterial color={p.skin} roughness={0.5} />
        </mesh>
        <mesh position={[0, 0.025, -0.094]}>
          <boxGeometry args={[0.12, 0.018, 0.02]} />
          <meshStandardMaterial color={p.front} emissive={p.front} emissiveIntensity={1.2} toneMapped={false} />
        </mesh>
        {/* ears (receiver positions) */}
        {[-1, 1].map((s) => (
          <mesh key={s} position={[s * 0.103, 0, 0]} scale={[0.4, 1, 0.7]}>
            <sphereGeometry args={[0.03, 12, 12]} />
            <meshStandardMaterial color={p.skin} roughness={0.6} />
          </mesh>
        ))}
        <Headphones p={p} />
      </group>
    </group>
  );
}
