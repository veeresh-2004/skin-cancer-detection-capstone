import { useEffect, useRef, useState } from "react";

function DatasetDistributionChart() {
  const [animated, setAnimated] = useState(false);
  const ref = useRef(null);

  const data = [
    { label: 'Benign', count: 4545, color: '#10b981', pct: 50 },
    { label: 'Melanoma', count: 4545, color: '#ef4444', pct: 50 },
  ];

  const total = 9090;
  const radius = 70;
  const stroke = 18;
  const cx = 90;
  const cy = 90;
  const circumference = 2 * Math.PI * radius;

  // Observer to trigger animation on scroll into view
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setAnimated(true); },
      { threshold: 0.3 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  // Build donut segments
  let offset = 0;
  const segments = data.map((d) => {
    const dash = animated ? (d.pct / 100) * circumference : 0;
    const gap = circumference - dash;
    const rotate = offset * 3.6 - 90; // -90 starts from top
    const seg = { ...d, dash, gap, rotate };
    offset += d.pct;
    return seg;
  });

  return (
    <div ref={ref} style={{ display: 'flex', alignItems: 'center', gap: 40, flexWrap: 'wrap' }}>
      {/* Donut SVG */}
      <div style={{ position: 'relative', flexShrink: 0 }}>
        <svg width={180} height={180} viewBox="0 0 180 180">
          {/* Background ring */}
          <circle
            cx={cx} cy={cy} r={radius}
            fill="none"
            stroke="var(--border)"
            strokeWidth={stroke}
          />
          {segments.map((seg, i) => (
            <circle
              key={seg.label}
              cx={cx} cy={cy} r={radius}
              fill="none"
              stroke={seg.color}
              strokeWidth={stroke}
              strokeDasharray={`${seg.dash} ${seg.gap}`}
              strokeLinecap="round"
              style={{
                transform: `rotate(${seg.rotate}deg)`,
                transformOrigin: `${cx}px ${cy}px`,
                transition: `stroke-dasharray 1.1s cubic-bezier(.4,0,.2,1) ${i * 0.15}s`,
              }}
            />
          ))}
          {/* Center label */}
          <text x={cx} y={cy - 8} textAnchor="middle" style={{ fontSize: 22, fontWeight: 700, fill: 'var(--navy)', fontFamily: 'inherit' }}>
            {total.toLocaleString()}
          </text>
          <text x={cx} y={cy + 12} textAnchor="middle" style={{ fontSize: 11, fill: 'var(--slate)', fontFamily: 'inherit' }}>
            images
          </text>
        </svg>
      </div>

      {/* Legend + animated stat bars */}
      <div style={{ flex: 1, minWidth: 180 }}>
        {data.map((d, i) => (
          <div key={d.label} style={{ marginBottom: 22 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.85rem', fontWeight: 600, marginBottom: 8 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--navy)' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: d.color, display: 'inline-block', flexShrink: 0 }} />
                {d.label}
              </span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', color: d.color }}>
                {d.count.toLocaleString()} — 50%
              </span>
            </div>
            {/* Animated segmented pip bar */}
            <div style={{ display: 'flex', gap: 3 }}>
              {Array.from({ length: 20 }).map((_, j) => (
                <div
                  key={j}
                  style={{
                    flex: 1,
                    height: 8,
                    borderRadius: 4,
                    background: d.color,
                    opacity: animated ? 1 : 0,
                    transform: animated ? 'scaleY(1)' : 'scaleY(0)',
                    transition: `opacity 0.4s ease ${i * 0.2 + j * 0.03}s, transform 0.4s ease ${i * 0.2 + j * 0.03}s`,
                    transformOrigin: 'bottom',
                  }}
                />
              ))}
            </div>
          </div>
        ))}
        <div style={{
          marginTop: 6,
          padding: '8px 12px',
          borderRadius: 8,
          background: 'var(--border)',
          fontSize: '.78rem',
          color: 'var(--slate)',
          fontWeight: 500,
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}>
          <span style={{ fontSize: '1rem' }}>⚖️</span>
          Perfectly balanced — 50 / 50 class split
        </div>
      </div>
    </div>
  );
}
export default DatasetDistributionChart;