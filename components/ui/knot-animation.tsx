"use client";

import React, { useEffect, useRef, useState } from "react";

const PI = Math.PI;
const RAMP = ".,-~:;=!*#$@";

const GEO_WATCH_PALETTE = [
  "#00B8A9",
  "#5555FF",
  "#B2E600",
  "#FF7070",
  "#00A192",
  "#3333CC",
  "#8FC500",
  "#FF5555"
];

type Vec3 = { x: number; y: number; z: number };

const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const mul = (v: Vec3, s: number): Vec3 => ({ x: v.x * s, y: v.y * s, z: v.z * s });
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x
});

const norm = (v: Vec3): Vec3 => {
  const m = Math.sqrt(Math.max(dot(v, v), 1e-8));
  return mul(v, 1 / m);
};

type KnotAnimationProps = {
  color?: boolean;
  speedA?: number;
  speedB?: number;
  width?: number;
  height?: number;
  className?: string;
};

export const KnotAnimation = ({
  color = true,
  speedA = 0.04,
  speedB = 0.02,
  width = 68,
  height = 30,
  className = ""
}: KnotAnimationProps) => {
  const [frame, setFrame] = useState<React.ReactElement[]>([]);
  const aRef = useRef(0);
  const bRef = useRef(0);

  useEffect(() => {
    let raf = 0;
    let active = true;

    const renderFrame = (a: number, b: number) => {
      const w = width;
      const h = height;
      const screen: string[] = Array(w * h).fill(" ");
      const paletteIdx: number[] = Array(w * h).fill(-1);
      const zbuf: number[] = Array(w * h).fill(0);

      const light = norm({ x: -1, y: 1, z: -1 });
      const cA = Math.cos(a);
      const sA = Math.sin(a);
      const cB = Math.cos(b);
      const sB = Math.sin(b);

      let tubeIdx = 0;
      for (let u = 0; u < 2 * PI; u += 0.06, tubeIdx++) {
        const c2 = 2 * u;
        const c3 = 3 * u;
        const center: Vec3 = {
          x: Math.sin(u) + 2 * Math.sin(c2),
          y: Math.cos(u) - 2 * Math.cos(c2),
          z: -Math.sin(c3)
        };

        const tangent = norm({
          x: Math.cos(u) + 4 * Math.cos(c2),
          y: -Math.sin(u) + 4 * Math.sin(c2),
          z: -3 * Math.cos(c3)
        });

        const up = Math.abs(dot(tangent, { x: 0, y: 1, z: 0 })) < 0.99
          ? { x: 0, y: 1, z: 0 }
          : { x: 1, y: 0, z: 0 };

        const normal = norm(cross(tangent, up));
        const binormal = cross(tangent, normal);
        const radius = 0.3;
        const segColorIdx = tubeIdx % GEO_WATCH_PALETTE.length;

        for (let v = 0; v < 2 * PI; v += 0.2) {
          const cv = Math.cos(v);
          const sv = Math.sin(v);
          const offset = add(mul(normal, cv * radius), mul(binormal, sv * radius));
          const point = add(center, offset);

          const x1 = point.x;
          const y1 = point.y * cA - point.z * sA;
          const z1 = point.y * sA + point.z * cA;

          const x2 = x1 * cB + z1 * sB;
          const y2 = y1;
          const z2 = -x1 * sB + z1 * cB + 5;

          const invz = 1 / z2;
          const px = Math.floor(w / 2 + (w * 0.52) * x2 * invz);
          const py = Math.floor(h / 2 - (h * 0.8) * y2 * invz);

          if (px >= 0 && px < w && py >= 0 && py < h) {
            const idx = px + py * w;
            if (invz > zbuf[idx]) {
              zbuf[idx] = invz;

              const nx1 = offset.x;
              const ny1 = offset.y * cA - offset.z * sA;
              const nz1 = offset.y * sA + offset.z * cA;

              const nx2 = nx1 * cB + nz1 * sB;
              const ny2 = ny1;
              const nz2 = -nx1 * sB + nz1 * cB;

              const lum = Math.max(0, dot(norm({ x: nx2, y: ny2, z: nz2 }), light));
              const ci = Math.min(RAMP.length - 1, Math.floor(lum * (RAMP.length - 1)));
              screen[idx] = RAMP[ci];
              paletteIdx[idx] = segColorIdx;
            }
          }
        }
      }

      const lines: React.ReactElement[] = [];
      for (let y = 0; y < h; y++) {
        const line: React.ReactElement[] = [];
        for (let x = 0; x < w; x++) {
          const idx = x + y * w;
          const ch = screen[idx];
          if (ch === " ") {
            line.push(<span key={x}> </span>);
          } else if (color) {
            line.push(
              <span key={x} style={{ color: GEO_WATCH_PALETTE[paletteIdx[idx]] }}>
                {ch}
              </span>
            );
          } else {
            line.push(<span key={x}>{ch}</span>);
          }
        }
        lines.push(<div key={y}>{line}</div>);
      }

      if (active) {
        setFrame(lines);
      }
    };

    const tick = () => {
      aRef.current += speedA;
      bRef.current += speedB;
      renderFrame(aRef.current, bRef.current);
      raf = window.requestAnimationFrame(tick);
    };

    raf = window.requestAnimationFrame(tick);

    return () => {
      active = false;
      window.cancelAnimationFrame(raf);
    };
  }, [color, speedA, speedB, width, height]);

  return (
    <pre className={`font-mono text-[8px] leading-none whitespace-pre text-center ${className}`}>
      {frame}
    </pre>
  );
};
