import { useEffect, useRef } from "react";
import "./Scene.css";

const SKYLINE = Array.from({ length: 34 }, (_, i) => {
  const x = (i / 34) * 1600 + (Math.random() - 0.5) * 20;
  const w = 30 + Math.random() * 26;
  const h = 40 + Math.random() * 130;
  return { x, w, h, key: `b${i}` };
});

const TOWER_WINDOWS = (() => {
  const rows = 16;
  const cols = 5;
  const windows = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const lit = Math.random() < 0.22;
      windows.push({
        key: `w${r}-${c}`,
        x: 1195 + c * 26,
        y: 210 + r * 30,
        lit,
        delay: (Math.random() * 6).toFixed(2),
        dur: (2.5 + Math.random() * 3).toFixed(2),
      });
    }
  }
  return windows;
})();

const REFLECTIONS = Array.from({ length: 7 }, (_, i) => ({
  key: `r${i}`,
  x: 60 + Math.random() * 1480,
  y: 790 + Math.random() * 90,
  w: 40 + Math.random() * 90,
  delay: (Math.random() * 5).toFixed(2),
  dur: (3 + Math.random() * 3).toFixed(2),
}));

export default function Scene() {
  const sceneRef = useRef(null);

  useEffect(() => {
    let raf = null;
    function handleMove(e) {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        const el = sceneRef.current;
        if (el) {
          const x = e.clientX / window.innerWidth - 0.5;
          const y = e.clientY / window.innerHeight - 0.5;
          el.style.setProperty("--mx", x.toFixed(4));
          el.style.setProperty("--my", y.toFixed(4));
        }
        raf = null;
      });
    }
    window.addEventListener("pointermove", handleMove);
    return () => window.removeEventListener("pointermove", handleMove);
  }, []);

  return (
    <div className="scene" ref={sceneRef} aria-hidden="true">
      <div className="scene-inner">
        <div className="layer layer-glow" />

        <svg
          className="layer layer-skyline"
          viewBox="0 0 1600 900"
          preserveAspectRatio="xMidYMax slice"
        >
          {SKYLINE.map((b) => (
            <rect
              key={b.key}
              x={b.x}
              y={760 - b.h}
              width={b.w}
              height={b.h}
              fill="#0d2547"
            />
          ))}
        </svg>

        <svg
          className="layer layer-bridge"
          viewBox="0 0 1600 900"
          preserveAspectRatio="xMidYMax slice"
        >
          <line x1="120" y1="700" x2="1300" y2="700" stroke="#cda36a" strokeWidth="1.5" opacity="0.5" />
          <line x1="380" y1="560" x2="380" y2="760" stroke="#cda36a" strokeWidth="3" opacity="0.55" />
          <line x1="960" y1="560" x2="960" y2="760" stroke="#cda36a" strokeWidth="3" opacity="0.55" />
          {Array.from({ length: 9 }).map((_, i) => (
            <line
              key={`cl${i}`}
              x1={380}
              y1={575}
              x2={220 + i * 46}
              y2={700}
              stroke="#cda36a"
              strokeWidth="0.8"
              opacity="0.3"
            />
          ))}
          {Array.from({ length: 9 }).map((_, i) => (
            <line
              key={`cr${i}`}
              x1={960}
              y1={575}
              x2={1120 + i * 20}
              y2={700}
              stroke="#cda36a"
              strokeWidth="0.8"
              opacity="0.3"
            />
          ))}
        </svg>

        <svg
          className="layer layer-water"
          viewBox="0 0 1600 900"
          preserveAspectRatio="xMidYMax slice"
        >
          <defs>
            <linearGradient id="waterGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0a1a35" stopOpacity="0.2" />
              <stop offset="100%" stopColor="#050d1c" stopOpacity="0.85" />
            </linearGradient>
          </defs>
          <rect x="0" y="760" width="1600" height="140" fill="url(#waterGrad)" />
          {REFLECTIONS.map((r) => (
            <rect
              key={r.key}
              className="reflection"
              x={r.x}
              y={r.y}
              width={r.w}
              height="2.5"
              rx="1.25"
              fill="#e3cba0"
              style={{ animationDelay: `${r.delay}s`, animationDuration: `${r.dur}s` }}
            />
          ))}
        </svg>

        <svg
          className="layer layer-tower"
          viewBox="0 0 1600 900"
          preserveAspectRatio="xMidYMax slice"
        >
          <polygon points="1160,760 1180,220 1330,190 1360,760" fill="#0e2a52" opacity="0.9" />
          <polygon points="1330,190 1360,760 1460,760 1420,240" fill="#0a2144" opacity="0.95" />
          <circle cx="1255" cy="192" r="4" className="beacon" fill="#e3cba0" />
          {TOWER_WINDOWS.map((w) => (
            <rect
              key={w.key}
              x={w.x}
              y={w.y}
              width="10"
              height="14"
              rx="1"
              className={w.lit ? "tower-window lit" : "tower-window"}
              style={
                w.lit
                  ? { animationDelay: `${w.delay}s`, animationDuration: `${w.dur}s` }
                  : undefined
              }
            />
          ))}
        </svg>
      </div>
    </div>
  );
}
