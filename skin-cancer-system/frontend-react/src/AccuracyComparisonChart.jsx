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

 function AccuracyComparisonChart() {
  const [ref, visible] = useInView();

  const metrics = [
    { label: 'Overall Accuracy', cnn: 79, hybrid: 87 },
    { label: 'Precision',        cnn: 81, hybrid: 90 },
    { label: 'Recall',           cnn: 78, hybrid: 88 },
    { label: 'F1 Score',         cnn: 79, hybrid: 89 },
  ];

  const R = 36, SW = 7, CX = 44, CY = 44, SIZE = 88;
  const circ = 2 * Math.PI * R;

  function GaugeRing({ value, color, delay }) {
    const dash = visible ? (value / 100) * circ : 0;
    return (
      <circle
        cx={CX} cy={CY} r={R}
        fill="none"
        stroke={color}
        strokeWidth={SW}
        strokeLinecap="round"
        strokeDasharray={`${dash} ${circ}`}
        style={{
          transform: 'rotate(-90deg)',
          transformOrigin: `${CX}px ${CY}px`,
          transition: `stroke-dasharray 1.1s cubic-bezier(.4,0,.2,1) ${delay}s`,
        }}
      />
    );
  }

  return (
    <div ref={ref} style={{
      background: 'var(--white)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--r-lg)',
      padding: '28px',
      boxShadow: 'var(--shadow-sm)',
      marginBottom: 28,
    }}>
      <div style={{
        fontSize: '.8rem', fontWeight: 700, letterSpacing: '.06em',
        textTransform: 'uppercase', color: 'var(--slate)', marginBottom: 24,
      }}>
        Accuracy Comparison — CNN vs CNN + CLIP
      </div>

      {/* Gauge grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 20, marginBottom: 24 }}>
        {metrics.map(({ label, cnn, hybrid }, i) => {
          const innerR = R - SW - 3;
          const innerCirc = 2 * Math.PI * innerR;
          return (
            <div key={label} style={{
              display: 'flex', flexDirection: 'column', alignItems: 'center',
              padding: '18px 12px', borderRadius: 12,
              background: 'linear-gradient(135deg,#f8fafc 0%,#f1f5f9 100%)',
              border: '1px solid var(--border)',
              position: 'relative', overflow: 'hidden',
            }}>
              {/* Corner glow */}
              <div style={{
                position: 'absolute', top: -20, right: -20,
                width: 60, height: 60, borderRadius: '50%',
                background: 'rgba(20,184,166,.08)', filter: 'blur(16px)',
              }} />

              <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
                {/* Outer track */}
                <circle cx={CX} cy={CY} r={R} fill="none" stroke="#e2e8f0" strokeWidth={SW} />
                {/* CNN outer ring */}
                <GaugeRing value={cnn} color="#94a3b8" delay={i * 0.12} />
                {/* Inner track */}
                <circle cx={CX} cy={CY} r={innerR} fill="none" stroke="#e2e8f0" strokeWidth={SW - 2} />
                {/* Hybrid inner ring */}
                <circle
                  cx={CX} cy={CY} r={innerR}
                  fill="none"
                  stroke="var(--teal, #14b8a6)"
                  strokeWidth={SW - 2}
                  strokeLinecap="round"
                  strokeDasharray={`${visible ? (hybrid / 100) * innerCirc : 0} ${innerCirc}`}
                  style={{
                    transform: 'rotate(-90deg)',
                    transformOrigin: `${CX}px ${CY}px`,
                    transition: `stroke-dasharray 1.1s cubic-bezier(.4,0,.2,1) ${i * 0.12 + 0.15}s`,
                  }}
                />
                {/* Center text */}
                <text x={CX} y={CY - 4} textAnchor="middle" style={{
                  fontSize: 13, fontWeight: 800,
                  fill: 'var(--teal,#14b8a6)',
                  fontFamily: 'JetBrains Mono, monospace',
                }}>
                  {hybrid}%
                </text>
                <text x={CX} y={CY + 10} textAnchor="middle" style={{
                  fontSize: 9, fill: '#94a3b8', fontFamily: 'inherit',
                }}>
                  +{hybrid - cnn}pp
                </text>
              </svg>

              <div style={{
                fontSize: '.78rem', fontWeight: 600,
                color: 'var(--navy)', textAlign: 'center', marginTop: 8,
              }}>
                {label}
              </div>

              {/* Legend */}
              <div style={{ display: 'flex', gap: 10, marginTop: 6 }}>
                {[['#94a3b8', `CNN ${cnn}%`], ['var(--teal,#14b8a6)', `Hybrid ${hybrid}%`]].map(([c, t]) => (
                  <span key={t} style={{
                    fontSize: '.68rem', color: 'var(--slate)',
                    display: 'flex', alignItems: 'center', gap: 3,
                  }}>
                    <span style={{
                      width: 7, height: 7, borderRadius: '50%',
                      background: c, display: 'inline-block', flexShrink: 0,
                    }} />
                    {t}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary strip */}
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
        <div style={{
          fontSize: '.72rem', fontWeight: 600, letterSpacing: '.05em',
          textTransform: 'uppercase', color: 'var(--slate)', marginBottom: 10,
        }}>
          Average improvement across all metrics
        </div>
        {[
          { label: 'CNN only',  value: 79.25, color: 'linear-gradient(90deg,#94a3b8,#cbd5e1)', textColor: 'var(--slate)',          delay: 0.2,  bold: false },
          { label: 'CNN+CLIP',  value: 88.5,  color: 'linear-gradient(90deg,var(--teal,#14b8a6),#2dd4bf)', textColor: 'var(--teal,#14b8a6)', delay: 0.5, bold: true  },
        ].map(({ label, value, color, textColor, delay, bold }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
            <span style={{ fontSize: '.72rem', color: textColor, width: 64, textAlign: 'right', fontWeight: bold ? 700 : 400 }}>
              {label}
            </span>
            <div style={{ flex: 1, height: 10, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: visible ? `${value}%` : '0%',
                background: color,
                borderRadius: 99,
                transition: `width 1.2s cubic-bezier(.4,0,.2,1) ${delay}s`,
              }} />
            </div>
            <span style={{
              fontSize: '.78rem', fontFamily: 'JetBrains Mono, monospace',
              color: textColor, fontWeight: bold ? 700 : 400, width: 36,
            }}>
              {Math.round(value)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default AccuracyComparisonChart;