import { useEffect, useRef, useState } from "react";

function useInView(threshold = 0.25) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) setVisible(true); },
      { threshold }
    );
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return [ref, visible];
}

 function CLIPPromptChart() {
  const [ref, visible] = useInView();
  const [hovered, setHovered] = useState(null);

  const prompts = [
    { label: 'Early stage melanoma',        score: 0.73, color: '#ef4444' },
    { label: 'Intermediate stage melanoma', score: 0.35, color: '#f59e0b' },
    { label: 'Advanced stage melanoma',     score: 0.12, color: '#f97316' },
    { label: 'Benign skin lesion',          score: 0.24, color: '#10b981' },
  ];

  const R_RADAR = 60, CX_R = 80, CY_R = 80, N = prompts.length;

  function polarToXY(angle, r) {
    return [CX_R + r * Math.cos(angle), CY_R + r * Math.sin(angle)];
  }

  const angles = prompts.map((_, i) => (i / N) * 2 * Math.PI - Math.PI / 2);
  const gridLevels = [0.25, 0.5, 0.75, 1];

  const radarPoints = prompts
    .map((p, i) => polarToXY(angles[i], (visible ? p.score : 0) * R_RADAR))
    .map(([x, y]) => `${x},${y}`)
    .join(' ');

  return (
    <div ref={ref}>
      <p className="section-title" style={{ marginTop: 8 }}>CLIP Prompt Similarity Scores</p>
      <p className="section-sub">
        Example output for a melanoma-positive sample — scores represent cosine similarity to each clinical prompt.
      </p>

      <div style={{
        background: 'var(--white)', border: '1px solid var(--border)',
        borderRadius: 'var(--r-lg)', padding: '28px',
        boxShadow: 'var(--shadow-sm)', marginBottom: 48,
      }}>
        <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', alignItems: 'flex-start' }}>

          {/* Radar chart */}
          <div style={{ flexShrink: 0 }}>
            <div style={{
              fontSize: '.72rem', fontWeight: 700, letterSpacing: '.06em',
              textTransform: 'uppercase', color: 'var(--slate)', marginBottom: 12,
            }}>
              Similarity Radar
            </div>
            <svg width={160} height={160} viewBox="0 0 160 160">
              {/* Grid rings */}
              {gridLevels.map((lvl) => (
                <polygon
                  key={lvl}
                  points={angles.map(a => polarToXY(a, lvl * R_RADAR)).map(([x, y]) => `${x},${y}`).join(' ')}
                  fill="none" stroke="#e2e8f0" strokeWidth={1}
                />
              ))}
              {/* Axis lines */}
              {angles.map((a, i) => {
                const [x, y] = polarToXY(a, R_RADAR);
                return <line key={i} x1={CX_R} y1={CY_R} x2={x} y2={y} stroke="#e2e8f0" strokeWidth={1} />;
              })}
              {/* Data fill */}
              <polygon
                points={radarPoints}
                fill="rgba(239,68,68,.12)"
                stroke="#ef4444"
                strokeWidth={2}
                strokeLinejoin="round"
                style={{ transition: 'points 1.2s cubic-bezier(.4,0,.2,1) 0.2s' }}
              />
              {/* Dots */}
              {prompts.map((p, i) => {
                const [x, y] = polarToXY(angles[i], (visible ? p.score : 0) * R_RADAR);
                return (
                  <circle
                    key={i} cx={x} cy={y} r={4}
                    fill={p.color} stroke="#fff" strokeWidth={2}
                    style={{ transition: `cx 1.2s ease ${i * 0.08}s, cy 1.2s ease ${i * 0.08}s` }}
                  />
                );
              })}
              {/* Score labels */}
              {prompts.map((p, i) => {
                const [x, y] = polarToXY(angles[i], R_RADAR + 14);
                return (
                  <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
                    style={{ fontSize: 8, fill: p.color, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace' }}>
                    {p.score.toFixed(2)}
                  </text>
                );
              })}
            </svg>
          </div>

          {/* Score bars */}
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{
              fontSize: '.72rem', fontWeight: 700, letterSpacing: '.06em',
              textTransform: 'uppercase', color: 'var(--slate)', marginBottom: 16,
            }}>
              Cosine Similarity Scores
            </div>

            {prompts.map((p, i) => {
              const isTop = p.score === Math.max(...prompts.map(x => x.score));
              const isHov = hovered === p.label;
              return (
                <div
                  key={p.label}
                  onMouseEnter={() => setHovered(p.label)}
                  onMouseLeave={() => setHovered(null)}
                  style={{
                    marginBottom: 14, cursor: 'default',
                    padding: '8px 10px', borderRadius: 10,
                    background: isHov ? `${p.color}0d` : 'transparent',
                    border: `1px solid ${isHov ? p.color + '33' : 'transparent'}`,
                    transition: 'background .2s, border-color .2s',
                  }}
                >
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    fontSize: '.83rem', marginBottom: 7, alignItems: 'center',
                  }}>
                    <span style={{ color: 'var(--navy)', fontWeight: 500, display: 'flex', alignItems: 'center', gap: 6 }}>
                      {isTop && (
                        <span style={{
                          fontSize: '.64rem', fontWeight: 700, background: p.color,
                          color: '#fff', borderRadius: 4, padding: '1px 5px', letterSpacing: '.04em',
                        }}>
                          TOP
                        </span>
                      )}
                      {`"${p.label}"`}
                    </span>
                    <span style={{
                      fontFamily: 'JetBrains Mono, monospace',
                      fontWeight: 700, color: p.color, fontSize: '.9rem',
                    }}>
                      {p.score.toFixed(2)}
                    </span>
                  </div>

                  {/* Glowing fill bar */}
                  <div style={{ position: 'relative', height: 10, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{
                      position: 'absolute', left: 0, top: 0, height: '100%',
                      width: visible ? `${p.score * 100}%` : '0%',
                      background: `linear-gradient(90deg, ${p.color}cc, ${p.color})`,
                      borderRadius: 99,
                      boxShadow: `0 0 8px ${p.color}88`,
                      transition: `width 1s cubic-bezier(.4,0,.2,1) ${i * 0.15}s`,
                    }} />
                  </div>

                  {/* Tick marks */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
                    {[0, 0.25, 0.5, 0.75, 1].map(t => (
                      <span key={t} style={{
                        fontSize: '.6rem', color: '#cbd5e1',
                        fontFamily: 'JetBrains Mono, monospace',
                      }}>
                        {t.toFixed(2)}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}

            <p style={{ fontSize: '.78rem', color: 'var(--slate)', marginTop: 8 }}>
              Highest score determines stage estimation shown in the prediction report.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CLIPPromptChart;