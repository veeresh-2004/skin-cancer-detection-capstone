import React, { useRef, useState, useEffect } from 'react'
import { jsPDF } from 'jspdf'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || '/predict'

// ── Inject global styles ──────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&family=Playfair+Display:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --navy:      #0a1628;
    --navy-mid:  #112240;
    --teal:      #0d9488;
    --teal-lt:   #14b8a6;
    --teal-pale: #ccfbf1;
    --amber:     #f59e0b;
    --red:       #ef4444;
    --red-pale:  #fee2e2;
    --slate:     #64748b;
    --slate-lt:  #94a3b8;
    --white:     #ffffff;
    --offwhite:  #f8fafc;
    --border:    #e2e8f0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,.08);
    --shadow-md: 0 4px 16px rgba(0,0,0,.1);
    --shadow-lg: 0 12px 40px rgba(0,0,0,.14);
    --r-sm: 8px;
    --r-md: 14px;
    --r-lg: 20px;
  }

  html { scroll-behavior: smooth; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--offwhite);
    color: var(--navy);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Nav ── */
  .nav {
    position: sticky; top: 0; z-index: 100;
    background: var(--navy);
    padding: 8px 32px;
    display: flex; align-items: center; min-height: 64px;
    box-shadow: 0 2px 20px rgba(0,0,0,.25);
  }
  .nav-logo {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem; font-weight: 600;
    color: var(--white); letter-spacing: .01em; margin-right: 32px;
  }
  .nav-logo .cross { color: var(--teal-lt); font-size: 1.3rem; }
  .nav-links { display: flex; gap: 4px; }
  .nav-btn {
    background: transparent; border: none; cursor: pointer;
    color: var(--slate-lt); font-family: 'DM Sans', sans-serif;
    font-size: .88rem; font-weight: 500; padding: 6px 14px;
    border-radius: 6px; transition: color .2s, background .2s;
    letter-spacing: .02em;
  }
  .nav-btn:hover { color: var(--white); background: rgba(255,255,255,.08); }
  .nav-btn.active { color: var(--teal-lt); background: rgba(13,148,136,.15); }

    /* Centered pill with circular buttons */
    .nav-center {
      position: absolute; left: 50%; top: 50%;
      transform: translate(-50%, -50%);
      display: flex; align-items: center; justify-content: center; pointer-events: none;
      width: calc(100% - 160px); max-width: 560px; justify-content: center;
    }
    .nav-pill {
      pointer-events: auto;
      display: inline-flex; align-items: center; gap: 8px;
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
      padding: 6px; border-radius: 999px; backdrop-filter: blur(6px);
      box-shadow: 0 6px 18px rgba(2,6,23,0.35);
      transition: transform .25s ease, box-shadow .25s ease;
    }
    .nav-pill:hover { transform: translateY(-3px); box-shadow: 0 10px 26px rgba(2,6,23,0.45); }
    .nav-pill .nav-btn {
      width: 44px; height: 44px; min-width: 44px; padding: 0;
      border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
      background: transparent; border: none; color: rgba(255,255,255,0.9);
      font-weight: 700; cursor: pointer; transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
      box-shadow: none;
    }
    .nav-pill .nav-btn .nav-label { display: none; margin-left: 8px; font-weight: 600; color: rgba(255,255,255,0.95); }
    .nav-pill .nav-btn:hover { transform: translateY(-4px) scale(1.05); box-shadow: 0 6px 14px rgba(2,6,23,0.35); }
    .nav-pill .nav-btn.active { background: linear-gradient(135deg, var(--teal), var(--teal-lt)); color: #fff; box-shadow: 0 8px 18px rgba(13,148,136,0.28); }

    /* large screens: show small label next to active/hovered button */
    @media(min-width: 900px) {
      .nav-pill .nav-btn .nav-label { display: inline-block; font-size: .86rem; }
      .nav-pill .nav-btn { width: auto; min-width: 44px; padding: 0 12px; border-radius: 999px; }
      .nav-pill { padding: 8px 10px; }
    }

    /* per-button container for mobile labels */
    .nav-item { position: relative; display: inline-flex; flex-direction: column; align-items: center; }
    .mobile-label {
      display: none; /* hidden on desktop */
      position: absolute; left: 50%; transform: translateX(-50%) translateY(-6px) scale(.98);
      top: calc(100% + 8px);
      background: rgba(239, 231, 231, 0.96); color: var(--vyo);
      padding: 6px 10px; border-radius: 8px; font-size: .9rem;
      border: 1px solid rgba(255,255,255,.06);
      box-shadow: 0 6px 14px rgba(2,6,23,0.22);
      opacity: 0; pointer-events: none;
      transition: opacity .22s ease, transform .22s ease;
      white-space: nowrap;
    }
    .mobile-label.show { opacity: 1; transform: translateX(-50%) translateY(0) scale(1); }
    @media (max-width: 720px) {
      .mobile-label { display: block; }
      /* keep the pill buttons compact (initials) while labels show below */
      .nav-pill .nav-btn { width: 44px; padding: 0; border-radius: 50%; }
      .nav-pill .nav-btn .nav-label { display: none; }
    }

      /* mobile: show current route below the pill */
      .nav-current {
        display: none;
        margin-top: 8px; text-align: center;
        font-size: .9rem; color: var(--white);
        background: rgba(255,255,255,.04); padding: 6px 10px; border-radius: 8px;
        border: 1px solid rgba(255,255,255,.06); box-shadow: 0 6px 14px rgba(2,6,23,0.25);
        transform-origin: top center; opacity: 0; transform: translateY(-6px) scale(.98);
        transition: opacity .25s ease, transform .25s ease;
      }
      @media (max-width: 720px) {
        .nav-current { display: block; opacity: 1; transform: translateY(0) scale(1); }
      }

  /* ── Animated page wrapper ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .page-enter { animation: fadeUp .45s ease both; }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 60%, #0f3460 100%);
    padding: 80px 48px 60px;
    position: relative; overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 60% at 80% 50%, rgba(13,148,136,.18), transparent);
    pointer-events: none;
  }
  .hero-grid {
    position: absolute; inset: 0; opacity: .04;
    background-image: linear-gradient(var(--teal) 1px, transparent 1px), linear-gradient(90deg, var(--teal) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }
  .hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(13,148,136,.2); border: 1px solid rgba(20,184,166,.35);
    color: var(--teal-lt); font-size: .78rem; font-weight: 600;
    letter-spacing: .08em; text-transform: uppercase;
    padding: 4px 12px; border-radius: 999px; margin-bottom: 20px;
  }
  .hero-tag .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--teal-lt); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.4;} }
  .hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 4vw, 3rem); font-weight: 700;
    color: var(--white); line-height: 1.2; max-width: 680px;
    margin-bottom: 16px;
  }
  .hero h1 em { color: var(--teal-lt); font-style: normal; }
  .hero-sub {
    color: var(--slate-lt); font-size: 1.05rem; max-width: 560px;
    line-height: 1.65; margin-bottom: 32px;
  }
  .hero-btns { display: flex; gap: 12px; flex-wrap: wrap; }
  .btn-primary {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--teal); color: var(--white);
    font-family: 'DM Sans', sans-serif; font-size: .9rem; font-weight: 600;
    border: none; cursor: pointer; padding: 12px 24px;
    border-radius: var(--r-sm); letter-spacing: .02em;
    transition: background .2s, transform .15s, box-shadow .2s;
    box-shadow: 0 4px 14px rgba(13,148,136,.4);
  }
  .btn-primary:hover { background: var(--teal-lt); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(13,148,136,.5); }
  .btn-outline {
    display: inline-flex; align-items: center; gap: 8px;
    background: transparent; color: var(--white);
    font-family: 'DM Sans', sans-serif; font-size: .9rem; font-weight: 500;
    border: 1px solid rgba(255,255,255,.25); cursor: pointer; padding: 12px 24px;
    border-radius: var(--r-sm); letter-spacing: .02em;
    transition: background .2s, border-color .2s;
  }
  .btn-outline:hover { background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.5); }

  /* ── Stats strip ── */
  .stats-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1px; background: var(--border);
    border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
  }
  .stat-cell {
    background: var(--white); padding: 22px 28px;
    display: flex; flex-direction: column; gap: 4px;
    animation: fadeUp .5s ease both;
  }
  .stat-cell:nth-child(2){animation-delay:.1s} .stat-cell:nth-child(3){animation-delay:.2s} .stat-cell:nth-child(4){animation-delay:.3s}
  .stat-num { font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 700; color: var(--teal); }
  .stat-label { font-size: .8rem; color: var(--slate); font-weight: 500; letter-spacing: .02em; }

  /* ── Section cards ── */
  .section-wrap { max-width: 1100px; margin: 0 auto; padding: 48px 24px; }
  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 600; color: var(--navy);
    margin-bottom: 6px;
  }
  .section-sub { font-size: .92rem; color: var(--slate); margin-bottom: 28px; }

  .info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 40px; }
  .info-card {
    background: var(--white); border: 1px solid var(--border);
    border-radius: var(--r-md); padding: 24px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .2s, transform .2s;
    animation: fadeUp .5s ease both;
  }
  .info-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
  .info-card:nth-child(1){animation-delay:.05s} .info-card:nth-child(2){animation-delay:.1s}
  .info-card:nth-child(3){animation-delay:.15s} .info-card:nth-child(4){animation-delay:.2s}
  .card-icon { font-size: 1.6rem; margin-bottom: 10px; }
  .card-title { font-size: .95rem; font-weight: 700; color: var(--navy); margin-bottom: 6px; }
  .card-body { font-size: .86rem; color: var(--slate); line-height: 1.6; }

  /* ── Steps ── */
  .steps { display: flex; flex-direction: column; gap: 12px; }
  .step {
    display: flex; align-items: flex-start; gap: 14px;
    background: var(--white); border: 1px solid var(--border);
    border-radius: var(--r-sm); padding: 16px 20px;
    animation: fadeUp .4s ease both;
  }
  .step-num {
    min-width: 30px; height: 30px; border-radius: 50%;
    background: var(--teal); color: var(--white);
    font-size: .82rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
  }
  .step-text { font-size: .9rem; color: var(--navy); line-height: 1.5; }
  .step-text strong { display: block; font-weight: 600; margin-bottom: 2px; }

  /* ── Detection form ── */
  .detect-layout { display: grid; grid-template-columns: minmax(220px,380px) 1fr; gap: 24px; align-items: start; }
  @media(max-width:900px){ .detect-layout { grid-template-columns: 1fr; } }

  .form-card {
    background: var(--white); border: 1px solid var(--border);
    border-radius: var(--r-lg); padding: 28px;
    box-shadow: var(--shadow-md);
    animation: fadeUp .4s ease both;
  }
  .form-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; font-weight: 600; color: var(--navy);
    margin-bottom: 20px; display: flex; align-items: center; gap: 8px;
  }
  .form-title span { color: var(--teal); }

  .field-group { display: flex; flex-direction: column; gap: 14px; }
  .field-row { display: flex; gap: 12px; }
  .field { display: flex; flex-direction: column; gap: 5px; flex: 1; }
  .field label { font-size: .78rem; font-weight: 600; color: var(--slate); letter-spacing: .04em; text-transform: uppercase; }
  .field input, .field select {
    padding: 9px 12px; border: 1.5px solid var(--border);
    border-radius: var(--r-sm); font-family: 'DM Sans', sans-serif;
    font-size: .9rem; color: var(--navy);
    background: var(--offwhite);
    transition: border-color .2s, box-shadow .2s; outline: none;
  }
  .field input:focus, .field select:focus {
    border-color: var(--teal); box-shadow: 0 0 0 3px rgba(13,148,136,.12);
    background: var(--white);
  }

  .sym-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .sym-label {
    display: flex; align-items: center; gap: 8px;
    font-size: .85rem; color: var(--navy); cursor: pointer;
    padding: 7px 10px; border-radius: var(--r-sm);
    border: 1.5px solid var(--border); background: var(--offwhite);
    transition: border-color .2s, background .2s;
    user-select: none;
  }
  .sym-label:hover { border-color: var(--teal-lt); background: var(--teal-pale); }
  .sym-label.checked { border-color: var(--teal); background: var(--teal-pale); color: var(--teal); font-weight: 600; }
  .sym-label input { display: none; }

  .upload-zone {
    border: 2px dashed var(--border); border-radius: var(--r-md);
    padding: 24px; text-align: center; cursor: pointer;
    background: var(--offwhite); transition: border-color .2s, background .2s;
    margin-top: 4px;
  }
  .upload-zone:hover { border-color: var(--teal); background: var(--teal-pale); }
  .upload-icon { font-size: 2rem; margin-bottom: 8px; }
  .upload-hint { font-size: .84rem; color: var(--slate); }
  .upload-hint strong { color: var(--teal); }

  .btn-analyze {
    width: 100%; margin-top: 16px;
    padding: 13px; font-size: .95rem; font-weight: 600;
    background: linear-gradient(135deg, var(--teal) 0%, #0f766e 100%);
    color: var(--white); border: none; border-radius: var(--r-sm);
    cursor: pointer; font-family: 'DM Sans', sans-serif;
    letter-spacing: .03em;
    transition: opacity .2s, transform .15s, box-shadow .2s;
    box-shadow: 0 4px 14px rgba(13,148,136,.35);
  }
  .btn-analyze:hover:not(:disabled) { opacity: .9; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(13,148,136,.45); }
  .btn-analyze:disabled { opacity: .55; cursor: not-allowed; }

  /* ── Results panel ── */
  .results-panel {
    display: flex; flex-direction: column; gap: 20px;
    animation: fadeUp .4s ease both;
  }

  .result-card {
    background: var(--white); border: 1px solid var(--border);
    border-radius: var(--r-lg); padding: 24px;
    box-shadow: var(--shadow-sm);
  }
  .result-card-title {
    font-size: .8rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; color: var(--slate);
    margin-bottom: 12px; display: flex; align-items: center; gap: 6px;
  }

  .prediction-badge {
    display: inline-flex; align-items: center; gap: 10px;
    padding: 10px 18px; border-radius: var(--r-sm);
    font-weight: 700; font-size: 1rem; margin-bottom: 10px;
  }
  .badge-benign { background: #d1fae5; color: #065f46; border: 1.5px solid #6ee7b7; }
  .badge-malignant { background: var(--red-pale); color: #991b1b; border: 1.5px solid #fca5a5; }
  .badge-unknown { background: #fef3c7; color: #92400e; border: 1.5px solid #fcd34d; }

  .conf-bar-wrap { margin-top: 10px; }
  .conf-bar-label { display: flex; justify-content: space-between; font-size: .8rem; color: var(--slate); margin-bottom: 4px; }
  .conf-bar-track { height: 6px; background: var(--border); border-radius: 99px; overflow: hidden; }
  .conf-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--teal), var(--teal-lt));
    transition: width .8s cubic-bezier(.4,0,.2,1);
  }

  .clip-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #eff6ff; border: 1.5px solid #bfdbfe;
    color: #1e40af; font-size: .83rem; font-weight: 600;
    padding: 6px 12px; border-radius: 99px;
  }

  .stage-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #fef2f2; border: 1.5px solid #fecaca;
    color: #b91c1c; font-size: .9rem; font-weight: 700;
    padding: 8px 16px; border-radius: var(--r-sm);
  }

  .error-box {
    background: #fef2f2; border: 1.5px solid #fecaca;
    border-radius: var(--r-sm); padding: 14px 18px;
    color: #991b1b; font-size: .88rem;
    display: flex; align-items: flex-start; gap: 8px;
  }

  .image-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media(max-width:560px){ .image-grid { grid-template-columns: 1fr; } }

  .img-box {
    background: var(--white); border: 1px solid var(--border);
    border-radius: var(--r-md); padding: 16px;
    box-shadow: var(--shadow-sm);
  }
  .img-box-title {
    font-size: .8rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; color: var(--slate); margin-bottom: 12px;
  }
  .img-placeholder {
    height: 180px; border: 2px dashed var(--border);
    border-radius: var(--r-sm); background: var(--offwhite);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 8px; color: var(--slate-lt); font-size: .85rem;
  }
  .img-box img {
    width: 100%; height: auto; border-radius: var(--r-sm);
    object-fit: contain; max-height: 60vh; max-width: 100%;
    animation: fadeUp .4s ease both;
  }

  /* ── Loading spinner ── */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    width: 20px; height: 20px; border: 2.5px solid rgba(255,255,255,.3);
    border-top-color: var(--white); border-radius: 50%;
    animation: spin .7s linear infinite; display: inline-block;
  }

  /* ── Skeleton ── */
  @keyframes shimmer {
    from { background-position: -200% 0; }
    to   { background-position: 200% 0; }
  }
  .skeleton {
    border-radius: 6px; height: 14px;
    background: linear-gradient(90deg, var(--border) 25%, #e8eef3 50%, var(--border) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
  }

  /* ── Download btn ── */
  .btn-download {
    display: flex; align-items: center; gap: 8px;
    padding: 11px 20px; border-radius: var(--r-sm);
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: var(--white); border: none; cursor: pointer;
    font-family: 'DM Sans', sans-serif; font-size: .88rem; font-weight: 600;
    transition: opacity .2s, transform .15s;
    box-shadow: 0 3px 12px rgba(37,99,235,.35);
  }
  .btn-download:hover:not(:disabled) { opacity: .9; transform: translateY(-1px); }
  .btn-download:disabled { opacity: .4; cursor: not-allowed; }

  /* ── Disclaimer ── */
  .disclaimer {
    background: #fffbeb; border: 1.5px solid #fde68a;
    border-radius: var(--r-sm); padding: 14px 18px;
    font-size: .82rem; color: #92400e; line-height: 1.6;
    display: flex; gap: 10px;
  }

  /* ── About ── */
  .about-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }
  .about-card {
    background: var(--white); border: 1px solid var(--border);
    border-radius: var(--r-md); padding: 24px;
    box-shadow: var(--shadow-sm);
  }
  .about-card h3 { font-family: 'Playfair Display', serif; font-size: 1rem; margin-bottom: 10px; color: var(--navy); }
  .about-card p { font-size: .87rem; color: var(--slate); line-height: 1.65; }

  /* ── Footer ── */
  .footer {
    background: var(--navy); color: var(--slate-lt);
    padding: 28px 32px; font-size: .82rem;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
    border-top: 1px solid rgba(255,255,255,.08);
  }
  .footer-logo { font-family: 'Playfair Display', serif; color: var(--white); font-size: .95rem; }

  /* ── Responsive tweaks (mobile-first) ── */
  @media (max-width: 900px) {
    .nav { padding: 0 16px; }
    .hero { padding: 56px 24px 40px; }
    .hero h1 { font-size: clamp(1.6rem, 5vw, 2.6rem); }
    .stats-strip { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
    .info-grid, .about-grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    .btn-primary, .btn-outline { padding: 10px 16px; font-size: .88rem; }
    .form-card { padding: 20px; }
    .img-placeholder { height: 160px; }
  }

  @media (max-width: 720px) {
    .nav-logo { font-size: 1rem; }
    .nav-links { gap: 6px; }
    .hero { padding: 48px 18px 32px; }
    .hero-sub { font-size: .95rem; }
    .detect-layout { grid-template-columns: 1fr; }
    .image-grid { grid-template-columns: 1fr; }
    .upload-zone { padding: 18px; }
    .stepper-responsive { grid-template-columns: 1fr !important; }
    .stat-num { font-size: 1.4rem; }
    /* hide the textual logo on mobile to avoid overlap with hero */
    .nav-logo { display: none; }
  }

  @media (max-width: 560px) {
    .nav { height: 56px; padding: 0 12px; }
    .nav-btn { padding: 6px 8px; font-size: .78rem; }
    .hero h1 { font-size: 1.6rem; }
    .hero-responsive { padding: 28px 16px 24px !important; }
    .info-card { padding: 16px; }
    .img-placeholder { height: 140px; }
    .btn-primary { width: 100%; }
    .skeleton { height: 12px; }
  }

  @media (max-width: 420px) {
    .nav-logo { font-size: .95rem; }
    .hero h1 { font-size: 1.4rem; }
    .hero-sub { font-size: .9rem; }
    .form-title { font-size: 1rem; }
    .field input, .field select { padding: 8px 10px; font-size: .86rem; }
    .img-placeholder { height: 120px; }
    .step { padding: 12px 14px; }
  }

  /* Team section styles */
  .team-guide-card {
    max-width: 920px; margin: 0 auto; padding: 18px; display:flex; gap:14px; align-items:center; border-radius:14px;
    background: var(--white); border: 1px solid var(--border); box-shadow: var(--shadow-sm);
  }
  .team-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 18px; max-width: 980px; margin: 0 auto; }
  .team-member { display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 12px; background: var(--white); border: 1px solid var(--border); border-radius: 12px; }
  .team-avatar { width: 96px; height: 96px; border-radius: 999px; position: relative; overflow: hidden; display:flex; align-items:center; justify-content:center; background: linear-gradient(135deg,var(--navy),var(--teal)); color: #fff; font-weight: 800; font-size: 1.1rem; }
  .team-avatar img { position: absolute; inset: 0; width: 100%; height: 121%; object-fit: cover; display: block; }

  @media (max-width: 720px) {
    .team-guide-card { padding: 14px; }
    .team-grid { grid-template-columns: repeat(2, 1fr); gap: 12px; }
  }
  @media (max-width: 420px) {
    .team-grid { grid-template-columns: 1fr; }
    .team-avatar { width: 72px; height: 72px; }
  }
`
function StyleInjector() {
  useEffect(() => {
    const el = document.createElement('style')
    el.textContent = GLOBAL_CSS
    document.head.appendChild(el)
    return () => document.head.removeChild(el)
  }, [])
  return null
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function getBadgeClass(label) {
  if (!label) return 'badge-unknown'
  const l = label.toLowerCase()
  if (l.includes('melanoma') || l.includes('malignant')) return 'badge-malignant'
  if (l.includes('benign') || l.includes('nevus') || l.includes('normal')) return 'badge-benign'
  return 'badge-unknown'
}

// ── PDF generation ────────────────────────────────────────────────────────────
function generatePDF({ name, age, gender, location, duration, symptoms, resultText, clipStatus, stage, previewSrc, gradcamSrc }) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' })
  const W = doc.internal.pageSize.getWidth()
  const H = doc.internal.pageSize.getHeight()
  const margin = 48

  // ── Header bar ──
  doc.setFillColor(10, 22, 40)
  doc.rect(0, 0, W, 72, 'F')
  doc.setFillColor(13, 148, 136)
  doc.rect(0, 72, W, 3, 'F')

  // logo cross
  doc.setTextColor(13, 148, 136)
  doc.setFontSize(22)
  doc.setFont('helvetica', 'bold')
  doc.text('✦', margin, 46)

  doc.setTextColor(255, 255, 255)
  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.text('Skin Cancer Detection Report', margin + 24, 46)

  doc.setTextColor(100, 116, 139)
  doc.setFontSize(9)
  doc.setFont('helvetica', 'normal')
  doc.text(`Generated: ${new Date().toLocaleString()}`, W - margin, 46, { align: 'right' })

  let y = 100

  // ── Patient section ──
  doc.setFillColor(248, 250, 252)
  doc.roundedRect(margin, y, W - margin * 2, 84, 6, 6, 'F')
  doc.setDrawColor(226, 232, 240)
  doc.roundedRect(margin, y, W - margin * 2, 84, 6, 6, 'S')

  doc.setTextColor(100, 116, 139)
  doc.setFontSize(8)
  doc.setFont('helvetica', 'bold')
  doc.text('PATIENT INFORMATION', margin + 16, y + 18)

  doc.setTextColor(10, 22, 40)
  doc.setFontSize(10)
  const col2 = margin + 180, col3 = margin + 330

  const fields = [
    ['Name', name || '—', margin + 16, y + 36],
    ['Age', age || '—', col2, y + 36],
    ['Gender', gender || '—', col3, y + 36],
    ['Lesion Location', location || '—', margin + 16, y + 58],
    ['Duration', duration || '—', col2, y + 58],
  ]
  fields.forEach(([lbl, val, x, fy]) => {
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(100, 116, 139)
    doc.setFontSize(8)
    doc.text(lbl.toUpperCase(), x, fy - 10)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(10, 22, 40)
    doc.setFontSize(10)
    doc.text(val, x, fy)
  })

  y += 100

  // ── Symptoms ──
  const activeSymptoms = Object.entries(symptoms || {}).filter(([, v]) => v).map(([k]) => k.replace('_', ' '))
  if (activeSymptoms.length > 0) {
    doc.setFillColor(240, 253, 250)
    doc.roundedRect(margin, y, W - margin * 2, 36, 6, 6, 'F')
    doc.setDrawColor(167, 243, 208)
    doc.roundedRect(margin, y, W - margin * 2, 36, 6, 6, 'S')
    doc.setTextColor(100, 116, 139)
    doc.setFontSize(8)
    doc.setFont('helvetica', 'bold')
    doc.text('REPORTED SYMPTOMS', margin + 16, y + 14)
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(9)
    doc.setTextColor(6, 78, 59)
    doc.text(activeSymptoms.join('  •  '), margin + 16, y + 27)
    y += 50
  }

  // ── Prediction ──
  const lines = (resultText || '').split('\n')
  const labelLine = lines.find(l => l.toLowerCase().includes('prediction:')) || ''
  const confLine = lines.find(l => l.toLowerCase().includes('confidence:')) || ''
  const predLabel = labelLine.replace(/^prediction:\s*/i, '')
  const isMalignant = predLabel.toLowerCase().includes('melanoma') || predLabel.toLowerCase().includes('malignant')

  doc.setFillColor(isMalignant ? 254 : 240, isMalignant ? 226 : 253, isMalignant ? 226 : 250)
  doc.roundedRect(margin, y, W - margin * 2, 76, 6, 6, 'F')
  doc.setDrawColor(isMalignant ? 252 : 167, isMalignant ? 165 : 243, isMalignant ? 165 : 208)
  doc.roundedRect(margin, y, W - margin * 2, 76, 6, 6, 'S')

  doc.setTextColor(100, 116, 139)
  doc.setFontSize(8)
  doc.setFont('helvetica', 'bold')
  doc.text('AI PREDICTION RESULT', margin + 16, y + 16)

  doc.setTextColor(isMalignant ? 185 : 6, isMalignant ? 28 : 95, isMalignant ? 28 : 70)
  doc.setFontSize(16)
  doc.setFont('helvetica', 'bold')
  doc.text(predLabel || resultText || 'N/A', margin + 16, y + 40)

  if (confLine) {
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(9)
    doc.setTextColor(71, 85, 105)
    doc.text(confLine, margin + 16, y + 58)
  }
  y += 92

  // ── CLIP & Stage ──
  if (clipStatus || stage) {
    const rowH = 42
    doc.setFillColor(239, 246, 255)
    doc.roundedRect(margin, y, W - margin * 2, rowH, 6, 6, 'F')
    doc.setDrawColor(191, 219, 254)
    doc.roundedRect(margin, y, W - margin * 2, rowH, 6, 6, 'S')
    doc.setTextColor(100, 116, 139)
    doc.setFontSize(8)
    doc.setFont('helvetica', 'bold')
    doc.text('CLIP VALIDATION', margin + 16, y + 14)
    if (stage) doc.text('ESTIMATED STAGE', margin + 250, y + 14)
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(10)
    doc.setTextColor(30, 64, 175)
    doc.text(clipStatus || '—', margin + 16, y + 30)
    if (stage) {
      doc.setTextColor(185, 28, 28)
      doc.setFont('helvetica', 'bold')
      doc.text(stage, margin + 250, y + 30)
    }
    y += 56
  }

  // ── Images ──
  const imgW = (W - margin * 2 - 20) / 2
  const imgH = 180
  const imgY = y + 10

  doc.setTextColor(100, 116, 139)
  doc.setFontSize(8)
  doc.setFont('helvetica', 'bold')
  doc.text('ORIGINAL IMAGE', margin, imgY - 8)
  doc.text('GRAD-CAM HEATMAP', margin + imgW + 20, imgY - 8)

  doc.setDrawColor(226, 232, 240)
  doc.roundedRect(margin, imgY, imgW, imgH, 6, 6, 'S')
  doc.roundedRect(margin + imgW + 20, imgY, imgW, imgH, 6, 6, 'S')

  if (previewSrc) {
    try { doc.addImage(previewSrc, 'JPEG', margin + 2, imgY + 2, imgW - 4, imgH - 4) }
    catch { try { doc.addImage(previewSrc, 'PNG', margin + 2, imgY + 2, imgW - 4, imgH - 4) } catch {} }
  }
  if (gradcamSrc) {
    try { doc.addImage(gradcamSrc, 'PNG', margin + imgW + 22, imgY + 2, imgW - 4, imgH - 4) } catch {}
  }
  y = imgY + imgH + 28

  // ── Disclaimer ──
  doc.setFillColor(255, 251, 235)
  doc.roundedRect(margin, y, W - margin * 2, 52, 6, 6, 'F')
  doc.setDrawColor(253, 230, 138)
  doc.roundedRect(margin, y, W - margin * 2, 52, 6, 6, 'S')
  doc.setTextColor(146, 64, 14)
  doc.setFontSize(8)
  doc.setFont('helvetica', 'bold')
  doc.text('⚠  DISCLAIMER', margin + 14, y + 16)
  doc.setFont('helvetica', 'normal')
  const disclaimer = 'This report is generated by an AI model and is intended for research and educational purposes only. It is not a substitute for professional medical diagnosis or advice. Please consult a qualified dermatologist or healthcare provider.'
  const dLines = doc.splitTextToSize(disclaimer, W - margin * 2 - 28)
  doc.text(dLines, margin + 14, y + 28)

  // ── Footer ──
  doc.setFillColor(10, 22, 40)
  doc.rect(0, H - 36, W, 36, 'F')
  doc.setTextColor(100, 116, 139)
  doc.setFontSize(8)
  doc.text('Skin Cancer Detection  ·  AI-Powered Medical Assistance', margin, H - 16)
  doc.text('CONFIDENTIAL — FOR EDUCATIONAL USE ONLY', W - margin, H - 16, { align: 'right' })

  const filename = `${(name || 'report').replace(/\s+/g, '_')}_skin_report.pdf`
  doc.save(filename)
}

// ── NavBar ────────────────────────────────────────────────────────────────────
function NavBar({ route, setRoute }) {
  return (
    <nav className="nav">
      <div className="nav-logo">
        <span className="cross">✦</span> DermoDetection
      </div>
      <div className="nav-center">
          <div className="nav-pill">
          {['home', 'detection', 'about', 'team'].map(r => (
            <div key={r} className="nav-item">
              <button
                className={`nav-btn${route === r ? ' active' : ''}`}
                onClick={() => setRoute(r)}
                aria-label={r}
              >
                <span className="nav-dot">{r.charAt(0).toUpperCase()}</span>
                <span className="nav-label">{r.charAt(0).toUpperCase() + r.slice(1)}</span>
              </button>
              <div className={`mobile-label${route === r ? ' show' : ''}`}>{r.charAt(0).toUpperCase() + r.slice(1)}</div>
            </div>
          ))}
        </div>
      </div>
    </nav>
  )
}

// ── Home ──────────────────────────────────────────────────────────────────────
function Home({ goToDetection }) {
  const [counters, setCounters] = useState({ acc: 0, time: 0, classes: 0 })
  const [activeStep, setActiveStep] = useState(0)

  useEffect(() => {
    const targets = { acc: 95, time: 3, classes: 7 }
    const duration = 1600
    const steps = 60
    let step = 0
    const timer = setInterval(() => {
      step++
      const ease = 1 - Math.pow(1 - step / steps, 3)
      setCounters({
        acc: Math.round(ease * targets.acc),
        time: +(ease * targets.time).toFixed(1),
        classes: Math.round(ease * targets.classes),
      })
      if (step >= steps) clearInterval(timer)
    }, duration / steps)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    const t = setInterval(() => setActiveStep(p => (p + 1) % 6), 2800)
    return () => clearInterval(t)
  }, [])

  const steps = [
    { icon: '📤', title: 'Upload Image', body: 'Submit a dermoscopic or standard skin lesion photograph via the Detection page.', color: '#0d9488' },
    { icon: '⚙️', title: 'Preprocessing', body: 'Image is resized to 224×224, normalized, and prepared for deep-learning inference.', color: '#6366f1' },
    { icon: '🧬', title: 'CNN Inference', body: 'EfficientNet extracts multi-scale features and computes classification probabilities.', color: '#f59e0b' },
    { icon: '🤖', title: 'CLIP Validation', body: 'Cross-checks CNN result against structured medical text prompt embeddings.', color: '#ec4899' },
    { icon: '🔥', title: 'Grad-CAM', body: 'Gradient heatmap overlaid on the lesion to highlight regions driving the decision.', color: '#ef4444' },
    { icon: '📄', title: 'Report', body: 'Download a structured clinical PDF with all findings, metrics, and disclaimers.', color: '#10b981' },
  ]

  return (
    <div className="page-enter">

      {/* ══════════════════════════════════════
          HERO + STATS
      ══════════════════════════════════════ */}
      <div style={{
        background: 'linear-gradient(135deg, #0a1628 0%, #112240 55%, #0f3460 100%)',
        position: 'relative', overflow: 'hidden',
      }}>
        {/* dot grid */}
        <div style={{
          position: 'absolute', inset: 0, opacity: .04,
          backgroundImage: 'radial-gradient(#14b8a6 1px, transparent 1px)',
          backgroundSize: '26px 26px', pointerEvents: 'none',
        }} />
        {/* ambient glow */}
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: 'radial-gradient(ellipse 50% 60% at 20% 60%, rgba(13,148,136,.14), transparent), radial-gradient(ellipse 40% 50% at 85% 30%, rgba(99,102,241,.1), transparent)',
        }} />

        {/* hero + stats grid */}
        <div style={{
          position: 'relative', zIndex: 1,
          display: 'grid',
          gridTemplateColumns: 'minmax(0,1.1fr) minmax(0,.9fr)',
          gap: 0, maxWidth: 1200, margin: '0 auto',
          padding: '64px 32px 56px',
        }}
          className="hero-responsive"
        >
          {/* ── LEFT: hero text ── */}
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', paddingRight: 48 }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              background: 'rgba(13,148,136,.18)', border: '1px solid rgba(20,184,166,.3)',
              color: '#5eead4', fontSize: '.72rem', fontWeight: 700,
              letterSpacing: '.1em', textTransform: 'uppercase',
              padding: '5px 14px', borderRadius: 999, marginBottom: 22, alignSelf: 'flex-start',
            }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#5eead4', animation: 'pulse 2s infinite', display: 'inline-block' }} />
              AI-Powered Dermatology
            </div>

            <h1 style={{
              fontFamily: 'Playfair Display, serif',
              fontSize: 'clamp(1.9rem, 3.2vw, 2.9rem)',
              fontWeight: 700, color: '#fff', lineHeight: 1.2, marginBottom: 18,
            }}>
              Early Melanoma<br />Detection with{' '}
              <span style={{
                backgroundImage: 'linear-gradient(90deg, #5eead4, #67e8f9)',
                WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent',
              }}>Explainable AI</span>
            </h1>

            <p style={{ color: '#94a3b8', fontSize: '.98rem', lineHeight: 1.78, maxWidth: 480, marginBottom: 32 }}>
              Upload a dermoscopic image for CNN-based classification,
              Grad-CAM visual explanations, and a structured clinical PDF report.
            </p>

            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <button className="btn-primary" onClick={goToDetection}>▶ Start Analysis</button>
              <button className="btn-outline"
                onClick={() => document.getElementById('learn')?.scrollIntoView({ behavior: 'smooth' })}>
                Learn More ↓
              </button>
            </div>

            {/* trust pills */}
            <div style={{ display: 'flex', gap: 8, marginTop: 28, flexWrap: 'wrap' }}>
              {['🔒 Privacy-first', '⚡ Fast results', '📄 PDF Reports', '🧬 HAM10000 Dataset'].map(p => (
                <span key={p} style={{
                  background: 'rgba(255,255,255,.06)', border: '1px solid rgba(255,255,255,.12)',
                  color: 'rgba(255,255,255,.6)', fontSize: '.73rem',
                  padding: '4px 11px', borderRadius: 999,
                }}>{p}</span>
              ))}
            </div>
          </div>

          {/* ── RIGHT: stats panel ── */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gridTemplateRows: 'auto auto',
            gap: 12,
            alignContent: 'center',
          }}>
            {[
              { num: `${counters.acc}%+`, label: 'Model Accuracy',   sub: 'ISIC test set',        icon: '🎯', color: '#14b8a6', span: false },
              { num: `<${counters.time}s`, label: 'Analysis Time',    sub: 'End-to-end',           icon: '⚡', color: '#67e8f9', span: false },
              { num: `${counters.classes}+`, label: 'Lesion Classes', sub: 'Dermoscopic types',    icon: '🔬', color: '#a78bfa', span: false },
              { num: 'XAI',              label: 'Explainable AI',     sub: 'Grad-CAM + CLIP',      icon: '🧠', color: '#fb923c', span: false },
            ].map(({ num, label, sub, icon, color }, i) => (
              <div key={label} style={{
                background: 'rgba(255,255,255,.05)',
                border: '1px solid rgba(255,255,255,.09)',
                borderRadius: 16, padding: '20px 18px',
                display: 'flex', flexDirection: 'column', gap: 10,
                animation: `fadeUp .5s ease both`,
                animationDelay: `${.15 + i * .1}s`,
                transition: 'background .25s, transform .2s',
                cursor: 'default',
              }}
                onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,.1)'; e.currentTarget.style.transform = 'translateY(-3px)' }}
                onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,.05)'; e.currentTarget.style.transform = 'translateY(0)' }}
              >
                {/* icon + label row */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '.72rem', fontWeight: 700, letterSpacing: '.06em', textTransform: 'uppercase', color: 'rgba(255,255,255,.45)' }}>{label}</span>
                  <div style={{
                    width: 32, height: 32, borderRadius: 8,
                    background: `${color}22`, border: `1px solid ${color}44`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '.95rem',
                  }}>{icon}</div>
                </div>
                {/* big number */}
                <div style={{
                  fontFamily: 'Playfair Display, serif',
                  fontSize: 'clamp(1.6rem, 2.5vw, 2rem)',
                  fontWeight: 700, color, lineHeight: 1,
                }}>{num}</div>
                {/* sub label */}
                <div style={{ fontSize: '.72rem', color: 'rgba(255,255,255,.38)', fontWeight: 500 }}>{sub}</div>
                {/* accent bar */}
                <div style={{ height: 2, background: `linear-gradient(90deg, ${color}, transparent)`, borderRadius: 99, marginTop: 2 }} />
              </div>
            ))}
          </div>
        </div>

        {/* responsive style tag */}
        <style>{`
          @media (max-width: 720px) {
            .hero-responsive {
              grid-template-columns: 1fr !important;
              padding: 40px 20px 36px !important;
            }
            .hero-responsive > div:first-child {
              padding-right: 0 !important;
              margin-bottom: 32px;
            }
          }
        `}</style>
      </div>

      {/* ══════════════════════════════════════
          LEARN SECTION
      ══════════════════════════════════════ */}
      <div className="section-wrap" id="learn">

        {/* ── Features ── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
          <div style={{ width: 4, height: 26, background: 'var(--teal)', borderRadius: 99 }} />
          <p className="section-title" style={{ margin: 0 }}>Why Early Detection Matters</p>
        </div>
        <p className="section-sub" style={{ paddingLeft: 16, marginBottom: 28 }}>
          Melanoma accounts for the majority of skin cancer deaths, yet 5-year survival rates exceed 98% when caught at Stage I.
        </p>

        <div className="info-grid" style={{ marginBottom: 56 }}>
          {[
            ['🔬', 'CNN Extraction',   'Deep convolutional layers extract hierarchical features from dermoscopic images for accurate classification.',     'var(--teal)', '#ccfbf1'],
            ['🧠', 'CLIP Alignment',   'CLIP aligns image embeddings with medical text prompts to validate predictions against clinical descriptions.',    '#6366f1',    '#ede9fe'],
            ['🌡️', 'Grad-CAM Maps',   'Gradient-weighted activation maps highlight exactly which lesion regions drove the model\'s decision.',            '#f59e0b',    '#fef3c7'],
            ['📄', 'Clinical Report',  'Download a structured PDF with prediction, confidence score, Grad-CAM overlay, and mandatory disclaimers.',        '#ec4899',    '#fce7f3'],
          ].map(([icon, title, body, color, bg]) => (
            <div key={title} className="info-card" style={{ borderTop: `3px solid ${color}`, overflow: 'hidden', position: 'relative' }}>
              <div style={{ position: 'absolute', top: -24, right: -24, width: 88, height: 88, background: bg, borderRadius: '50%', opacity: .6 }} />
              <div style={{ fontSize: '1.7rem', marginBottom: 12 }}>{icon}</div>
              <div style={{ fontSize: '.93rem', fontWeight: 700, color: 'var(--navy)', marginBottom: 7 }}>{title}</div>
              <div style={{ fontSize: '.84rem', color: 'var(--slate)', lineHeight: 1.65 }}>{body}</div>
            </div>
          ))}
        </div>

        {/* ══════════════════════════════════════
            HOW IT WORKS — interactive stepper
        ══════════════════════════════════════ */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
          <div style={{ width: 4, height: 26, background: 'var(--teal)', borderRadius: 99 }} />
          <p className="section-title" style={{ margin: 0 }}>How It Works</p>
        </div>
        <p className="section-sub" style={{ paddingLeft: 16, marginBottom: 32 }}>
          Six steps from image upload to clinical-grade report. Click any step to explore.
        </p>

        {/* stepper layout */}
        <div style={{
          background: 'var(--vyo)', border: '1px solid var(--border)',
          borderRadius: 20, overflow: 'hidden', boxShadow: 'var(--shadow-md)',
          marginBottom: 48,
          display: 'grid', gridTemplateColumns: '260px 1fr',
        }}
          className="stepper-responsive"
        >
          {/* left: step list */}
          <div style={{ borderRight: '1px solid var(--border)', background: 'var(--offwhite)' }}>
            {steps.map((s, i) => (
              <div key={s.title}
                onClick={() => setActiveStep(i)}
                style={{
                  padding: '16px 20px', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', gap: 12,
                  borderLeft: `3px solid ${i === activeStep ? s.color : 'transparent'}`,
                  background: i === activeStep ? `${s.color}0f` : 'transparent',
                  transition: 'all .25s',
                  borderBottom: i < steps.length - 1 ? '1px solid var(--border)' : 'none',
                }}
                onMouseEnter={e => { if (i !== activeStep) e.currentTarget.style.background = 'rgba(0,0,0,.03)' }}
                onMouseLeave={e => { if (i !== activeStep) e.currentTarget.style.background = 'transparent' }}
              >
                {/* step number bubble */}
                <div style={{
                  width: 30, height: 30, borderRadius: '50%', flexShrink: 0,
                  background: i === activeStep ? s.color : 'var(--border)',
                  color: i === activeStep ? '#fff' : 'var(--slate)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '.75rem', fontWeight: 800,
                  transition: 'background .25s, color .25s',
                  boxShadow: i === activeStep ? `0 0 0 4px ${s.color}28` : 'none',
                }}>{i + 1}</div>

                <div>
                  <div style={{ fontSize: '.85rem', fontWeight: i === activeStep ? 700 : 500, color: i === activeStep ? 'var(--navy)' : 'var(--slate)', transition: 'color .2s' }}>{s.title}</div>
                </div>
              </div>
            ))}
          </div>

          {/* right: step detail panel */}
          <div style={{ padding: '40px 44px', display: 'flex', flexDirection: 'column', justifyContent: 'center', minHeight: 340, position: 'relative', overflow: 'hidden' }}>
            {/* large background icon */}
            <div style={{
              position: 'absolute', right: 32, top: '50%', transform: 'translateY(-50%)',
              fontSize: '7rem', opacity: .5, userSelect: 'none', pointerEvents: 'none',
            }}>{steps[activeStep].icon}</div>

            {/* step badge */}
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 8,
              background: `${steps[activeStep].color}15`,
              border: `1px solid ${steps[activeStep].color}35`,
              color: steps[activeStep].color,
              fontSize: '.72rem', fontWeight: 700, letterSpacing: '.08em', textTransform: 'uppercase',
              padding: '5px 14px', borderRadius: 999, marginBottom: 20, alignSelf: 'flex-start',
            }}>
              <span style={{ fontSize: '1rem' }}>{steps[activeStep].icon}</span>
              Step {activeStep + 1} of {steps.length}
            </div>

            <h2 style={{
              fontFamily: 'Playfair Display, serif',
              fontSize: '1.65rem', fontWeight: 700,
              color: 'var(--navy)', marginBottom: 16, lineHeight: 1.25,
            }}>{steps[activeStep].title}</h2>

            <p style={{ fontSize: '.95rem', color: 'var(--slate)', lineHeight: 1.8, maxWidth: 420, marginBottom: 28 }}>
              {steps[activeStep].body}
            </p>

            {/* progress dots */}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {steps.map((s, i) => (
                <div key={i} onClick={() => setActiveStep(i)} style={{
                  width: i === activeStep ? 24 : 8,
                  height: 8, borderRadius: 99,
                  background: i === activeStep ? steps[activeStep].color : 'var(--border)',
                  cursor: 'pointer', transition: 'all .3s',
                }} />
              ))}
              <span style={{ marginLeft: 8, fontSize: '.72rem', color: 'var(--slate-lt)' }}>auto-advancing</span>
            </div>

            {/* prev / next */}
            <div style={{ display: 'flex', gap: 10, marginTop: 20 }}>
              <button onClick={() => setActiveStep(p => (p - 1 + 6) % 6)} style={{
                padding: '8px 16px', borderRadius: 8,
                border: '1.5px solid var(--border)', background: 'transparent',
                color: 'var(--slate)', fontSize: '.83rem', cursor: 'pointer',
                transition: 'border-color .2s',
              }}
                onMouseEnter={e => e.currentTarget.style.borderColor = steps[activeStep].color}
                onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
              >← Prev</button>
              <button onClick={() => setActiveStep(p => (p + 1) % 6)} style={{
                padding: '8px 20px', borderRadius: 8,
                border: 'none', background: steps[activeStep].color,
                color: '#fff', fontSize: '.83rem', fontWeight: 600, cursor: 'pointer',
                transition: 'opacity .2s',
              }}
                onMouseEnter={e => e.currentTarget.style.opacity = '.85'}
                onMouseLeave={e => e.currentTarget.style.opacity = '1'}
              >Next →</button>
            </div>
          </div>
        </div>

        <style>{`
          @media (max-width: 640px) {
            .stepper-responsive {
              grid-template-columns: 1fr !important;
            }
            .stepper-responsive > div:first-child {
              border-right: none !important;
              border-bottom: 1px solid var(--border);
              display: flex; flex-wrap: wrap;
            }
            .stepper-responsive > div:first-child > div {
              flex: 1 1 45%;
              border-bottom: none !important;
            }
          }
        `}</style>

        {/* Disclaimer */}
        <div className="disclaimer">
          <span>⚠️</span>
          <span><strong>Disclaimer:</strong> This tool is intended for research and educational purposes only and must not replace professional medical diagnosis. Always consult a qualified dermatologist for clinical decisions.</span>
        </div>
      </div>
    </div>
  )
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [route, setRoute] = useState('home')

  const fileInputRef = useRef(null)
  const [previewSrc, setPreviewSrc] = useState('')
  const [resultText, setResultText] = useState('')
  const [errorText, setErrorText] = useState('')
  const [clipStatus, setClipStatus] = useState('')
  const [stage, setStage] = useState(null)
  const [loading, setLoading] = useState(false)
  const [gradcamSrc, setGradcamSrc] = useState('')
  const [confidence, setConfidence] = useState(null)
  const [predLabel, setPredLabel] = useState('')

  const [name, setName] = useState('')
  const [age, setAge] = useState('')
  const [gender, setGender] = useState('')
  const [location, setLocation] = useState('Face')
  const [duration, setDuration] = useState('Less than 1 month')
  const [symptoms, setSymptoms] = useState({ itching: false, bleeding: false, pain: false, rapid_growth: false, color_change: false })

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => {
      setPreviewSrc(ev.target.result)
      setGradcamSrc(''); setResultText(''); setErrorText(''); setClipStatus(''); setStage(null); setConfidence(null); setPredLabel('')
    }
    reader.readAsDataURL(file)
  }

  function toggleSymptom(key) { setSymptoms(p => ({ ...p, [key]: !p[key] })) }

  async function handleAnalyzeClick() {
    const file = fileInputRef.current?.files?.[0]
    if (!file) { setErrorText('Please select an image first.'); return }
    setLoading(true); setResultText(''); setErrorText(''); setClipStatus(''); setStage(null); setGradcamSrc(''); setConfidence(null); setPredLabel('')

    const formData = new FormData()
    formData.append('image', file)
    formData.append('name', name); formData.append('age', age); formData.append('gender', gender)
    formData.append('location', location); formData.append('duration', duration)
    formData.append('symptoms', JSON.stringify(symptoms))

    try {
      const resp = await fetch(BACKEND_URL, { method: 'POST', body: formData, credentials: 'same-origin' })
      if (!resp.ok) {
        const raw = await resp.text()
        let msg = `Server error: ${resp.status}`
        try { const p = JSON.parse(raw); if (p.error) msg = p.error } catch {}
        setErrorText(msg); setLoading(false); return
      }
      const data = await resp.json()
      if (data.error) { setErrorText(data.error); setLoading(false); return }

      const frac = typeof data.confidence === 'number' ? data.confidence : (data.confidence_percent ? data.confidence_percent / 100 : null)
      const pct = frac !== null ? frac * 100 : null
      const label = data.label || ''
      setPredLabel(label)
      setConfidence(pct)
      setResultText(`Prediction: ${label}${pct !== null ? `\nConfidence: ${frac.toFixed(4)} (${pct.toFixed(2)}%)` : ''}`)
      setClipStatus(data.clip_validation || '')
      setStage(data.stage || null)
      if (data.gradcam_image) setGradcamSrc('data:image/png;base64,' + data.gradcam_image)
    } catch (err) {
      setErrorText(`Error: ${err?.message || err}`)
    } finally {
      setLoading(false)
    }
  }

  function handleDownloadReport() {
    generatePDF({ name, age, gender, location, duration, symptoms, resultText, clipStatus, stage, previewSrc, gradcamSrc })
  }

  const badgeClass = getBadgeClass(predLabel)

  return (
    <div>
      <StyleInjector />
      <NavBar route={route} setRoute={setRoute} />

      {/* ── HOME ── */}
      {route === 'home' && <Home goToDetection={() => setRoute('detection')} />}

      {/* ── DETECTION ── */}
      {route === 'detection' && (
        <div className="section-wrap page-enter">
          <p className="section-title">Skin Lesion Analysis</p>
          <p className="section-sub">Complete the patient form, upload a dermoscopic image, and receive an AI-powered assessment.</p>

          <div className="detect-layout">
            {/* Form */}
            <div className="form-card">
              <div className="form-title"><span>✦</span> Patient Details</div>
              <div className="field-group">
                <div className="field-row">
                  <div className="field">
                    <label>Full Name</label>
                    <input placeholder="e.g. Alex Johnson" value={name} onChange={e => setName(e.target.value)} />
                  </div>
                </div>
                <div className="field-row">
                  <div className="field" style={{ maxWidth: 100 }}>
                    <label>Age</label>
                    <input placeholder="35" value={age} onChange={e => setAge(e.target.value)} />
                  </div>
                  <div className="field">
                    <label>Gender</label>
                    <select value={gender} onChange={e => setGender(e.target.value)}>
                      <option value="">Select</option>
                      <option>Male</option><option>Female</option><option>Other</option>
                    </select>
                  </div>
                </div>
                <div className="field-row">
                  <div className="field">
                    <label>Lesion Location</label>
                    <select value={location} onChange={e => setLocation(e.target.value)}>
                      {['Face','Arm','Back','Leg','Chest','Other'].map(o => <option key={o}>{o}</option>)}
                    </select>
                  </div>
                  <div className="field">
                    <label>Duration</label>
                    <select value={duration} onChange={e => setDuration(e.target.value)}>
                      {['Less than 1 month','1–6 months','More than 6 months'].map(o => <option key={o}>{o}</option>)}
                    </select>
                  </div>
                </div>

                <div className="field">
                  <label>Symptoms</label>
                  <div className="sym-grid">
                    {[['itching','🔴 Itching'],['bleeding','💧 Bleeding'],['pain','⚡ Pain'],['rapid_growth','📈 Rapid Growth'],['color_change','🎨 Color Change']].map(([k, lbl]) => (
                      <label key={k} className={`sym-label${symptoms[k] ? ' checked' : ''}`} onClick={() => toggleSymptom(k)}>
                        <input type="checkbox" readOnly checked={symptoms[k]} /> {lbl}
                      </label>
                    ))}
                  </div>
                </div>

                <div className="field">
                  <label>Image Upload</label>
                  <label htmlFor="imageInput" className="upload-zone">
                    {previewSrc
                      ? <img src={previewSrc} alt="preview" style={{ maxHeight: 120, borderRadius: 8, objectFit: 'contain' }} />
                      : (<><div className="upload-icon">🔬</div><div className="upload-hint"><strong>Click to upload</strong> or drag and drop<br />JPEG, PNG, TIFF supported</div></>)
                    }
                  </label>
                  <input id="imageInput" ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
                </div>

                <button className="btn-analyze" onClick={handleAnalyzeClick} disabled={loading}>
                  {loading ? <><span className="spinner" /> Analyzing…</> : '▶ Analyze Image'}
                </button>
              </div>
            </div>

            {/* Results */}
            <div className="results-panel">
              {/* Loading skeleton */}
              {loading && (
                <div className="result-card">
                  <div className="result-card-title">⏳ Processing Analysis</div>
                  {[100, 75, 55].map(w => <div key={w} className="skeleton" style={{ width: `${w}%`, marginBottom: 10 }} />)}
                </div>
              )}

              {/* Error */}
              {errorText && !loading && (
                <div className="error-box"><span>⚠️</span><span>{errorText}</span></div>
              )}

              {/* Prediction */}
              {resultText && !loading && (
                <div className="result-card">
                  <div className="result-card-title">🧬 Prediction Result</div>
                  <div className={`prediction-badge ${badgeClass}`}>
                    {badgeClass === 'badge-malignant' ? '⚠️' : badgeClass === 'badge-benign' ? '✅' : '🔍'} {predLabel || resultText}
                  </div>
                  {confidence !== null && (
                    <div className="conf-bar-wrap">
                      <div className="conf-bar-label"><span>Confidence</span><span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 600 }}>{confidence.toFixed(2)}%</span></div>
                      <div className="conf-bar-track"><div className="conf-bar-fill" style={{ width: `${Math.min(confidence, 100)}%` }} /></div>
                    </div>
                  )}
                </div>
              )}

              {clipStatus && !loading && (
                <div className="result-card">
                  <div className="result-card-title">🤖 CLIP Validation</div>
                  <div className="clip-chip">🔵 {clipStatus}</div>
                </div>
              )}

              {stage && !loading && (
                <div className="result-card">
                  <div className="result-card-title">📊 Estimated Stage</div>
                  <div className="stage-badge">⚠️ {stage}</div>
                </div>
              )}

              {/* Images */}
              <div className="image-grid">
                <div className="img-box">
                  <div className="img-box-title">📷 Original Image</div>
                  {previewSrc
                    ? <img src={previewSrc} alt="Original" />
                    : <div className="img-placeholder"><span>📷</span>No image selected</div>
                  }
                </div>
                <div className="img-box">
                  <div className="img-box-title">🔥 Grad-CAM Heatmap</div>
                  {gradcamSrc
                    ? <img src={gradcamSrc} alt="Grad-CAM" />
                    : <div className="img-placeholder"><span>{loading ? '⏳' : '🔬'}</span>{loading ? 'Generating heatmap…' : 'Awaiting analysis'}</div>
                  }
                </div>
              </div>

              {/* Download */}
              {resultText && !loading && (
                <div style={{ display: 'flex', gap: 12 }}>
                  <button className="btn-download" onClick={handleDownloadReport}>
                    ⬇ Download PDF Report
                  </button>
                </div>
              )}

              {/* Disclaimer */}
              <div className="disclaimer">
                <span>⚠️</span>
                <span><strong>Research use only.</strong> This AI output is not a clinical diagnosis. Consult a board-certified dermatologist for medical advice.</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── ABOUT ── */}
      {route === 'about' && (
        <div className="page-enter">

          {/* ── About Hero ── */}
          <div style={{ background: 'linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%)', padding: '52px 48px 44px', borderBottom: '3px solid var(--teal)' }}>
            <div style={{ maxWidth: 700 }}>
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, background: 'rgba(13,148,136,.2)', border: '1px solid rgba(20,184,166,.3)', color: 'var(--teal-lt)', fontSize: '.78rem', fontWeight: 600, letterSpacing: '.08em', textTransform: 'uppercase', padding: '4px 12px', borderRadius: 999, marginBottom: 18 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--teal-lt)', display: 'inline-block' }} /> System Overview
              </div>
              <h1 style={{ fontFamily: 'Playfair Display, serif', fontSize: 'clamp(1.6rem,3vw,2.4rem)', fontWeight: 700, color: 'var(--white)', lineHeight: 1.25, marginBottom: 14 }}>About <span style={{ color: 'var(--teal-lt)' }}>DermoDetection</span></h1>
              <p style={{ color: 'var(--slate-lt)', fontSize: '1rem', lineHeight: 1.7 }}>A hybrid CNN + CLIP ensemble with Grad-CAM explainability for clinical-grade dermoscopic skin lesion classification. Built on the HAM10000 Archive — the world's largest public dermoscopy dataset.</p>
            </div>
          </div>

          <div className="section-wrap">

            {/* ── System cards ── */}
            <p className="section-title">System Architecture</p>
            <p className="section-sub">How the model pipeline is structured under the hood.</p>
            <div className="about-grid" style={{ marginBottom: 48 }}>
              {[
                
                ['🧬','Architecture','Combines an EfficientNet-based CNN for feature extraction with CLIPs cross-modal vision-language alignment. Grad-CAM generates pixel-level attribution maps.'],
                ['📊','Dataset','Trained and evaluated on the HAM10000 Archive with 9,090 balanced images — 4,545 benign and 4,545 melanoma — augmented to prevent class imbalance.'],
                ['🔐','Privacy','Images are processed server-side and deleted immediately after inference. No patient data is stored, logged, or retained.'],
                ['📄','Reports','PDF reports include patient demographics, prediction result, confidence score, CLIP validation, Grad-CAM overlay, and a mandatory clinical disclaimer.'],
                ['⚠️','Limitations','Model performs best on high-quality dermoscopic images. Results may degrade on standard smartphone photographs or images with poor lighting.'],
                ['📬','Contact','For research inquiries, dataset access, or collaboration requests, refer to the project README or repository documentation.'],
              ].map(([icon, title, body]) => (
                <div className="about-card" key={title}>
                  <h3>{icon} {title}</h3>
                  <p>{body}</p>
                </div>
              ))}
            </div>

            {/* ── Model Pipeline ── */}
            <p className="section-title">Model Pipeline</p>
            <p className="section-sub">Sequential CNN architecture — input to sigmoid output.</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 0, flexWrap: 'wrap', marginBottom: 48, background: 'var(--white)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', padding: '28px 24px', boxShadow: 'var(--shadow-sm)', overflowX: 'auto' }}>
              {[
                ['🖼️','Input','224×224 px'],
                ['⬛','Conv2D','32 filters'],
                ['⬇️','MaxPool','2×2'],
                ['⬛','Conv2D','64 filters'],
                ['⬇️','MaxPool','2×2'],
                ['📐','Flatten','—'],
                ['🔵','Dense','256 units'],
                ['💧','Dropout','0.5'],
                ['🎯','Sigmoid','Output'],
              ].map(([icon, name, detail], i, arr) => (
                <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, minWidth: 72 }}>
                    <div style={{ width: 52, height: 52, borderRadius: 12, background: name === 'Sigmoid' ? 'var(--teal)' : name === 'Input' ? 'var(--navy)' : 'var(--offwhite)', border: `2px solid ${name === 'Sigmoid' ? 'var(--teal)' : name === 'Input' ? 'var(--navy)' : 'var(--border)'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem', boxShadow: 'var(--shadow-sm)' }}>
                      {icon}
                    </div>
                    <span style={{ fontSize: '.72rem', fontWeight: 700, color: 'var(--navy)', letterSpacing: '.02em' }}>{name}</span>
                    <span style={{ fontSize: '.68rem', color: 'var(--slate-lt)' }}>{detail}</span>
                  </div>
                  {i < arr.length - 1 && <div style={{ width: 24, height: 2, background: 'var(--teal)', opacity: .4, margin: '0 2px', marginBottom: 20 }} />}
                </div>
              ))}
            </div>

            {/* ── Performance Metrics ── */}
            <p className="section-title">Performance Metrics</p>
            <p className="section-sub">Classification report on the held-out test set.</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px,1fr))', gap: 14, marginBottom: 48 }}>
              {[
                ['Accuracy','83%','Overall correct predictions','var(--teal)'],
                ['Precision (Melanoma)','90%','Of predicted melanoma, truly melanoma','#6366f1'],
                ['Recall (Melanoma)','88%','Melanoma cases correctly caught','#f59e0b'],
                ['F1 Score','89%','Harmonic mean of precision & recall','#10b981'],
                ['Precision (Benign)','87%','Of predicted benign, truly benign','#3b82f6'],
                ['Recall (Benign)','82%','Benign cases correctly identified','#ec4899'],
              ].map(([label, val, desc, color]) => (
                <div key={label} style={{ background: 'var(--white)', border: '1px solid var(--border)', borderRadius: 'var(--r-md)', padding: '20px 18px', boxShadow: 'var(--shadow-sm)', borderTop: `3px solid ${color}` }}>
                  <div style={{ fontFamily: 'Playfair Display, serif', fontSize: '1.8rem', fontWeight: 700, color }}>{val}</div>
                  <div style={{ fontSize: '.88rem', fontWeight: 700, color: 'var(--navy)', margin: '4px 0 4px' }}>{label}</div>
                  <div style={{ fontSize: '.78rem', color: 'var(--slate)', lineHeight: 1.5 }}>{desc}</div>
                </div>
              ))}
            </div>

            {/* ── Dataset Distribution ── */}
            <p className="section-title">Dataset Distribution</p>
            <p className="section-sub">Balanced training set — 9,090 images total from the HAM10000 Archive.</p>
            <div style={{ background: 'var(--white)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', padding: '28px', boxShadow: 'var(--shadow-sm)', marginBottom: 48 }}>
              {[['Benign', 4545, '#10b981'], ['Melanoma', 4545, '#ef4444']].map(([label, count, color]) => (
                <div key={label} style={{ marginBottom: 18 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.85rem', fontWeight: 600, marginBottom: 6 }}>
                    <span style={{ color: 'var(--navy)' }}>{label}</span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', color }}>{count.toLocaleString()} images — 50%</span>
                  </div>
                  <div style={{ height: 10, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{ width: '50%', height: '100%', background: color, borderRadius: 99, transition: 'width .8s ease' }} />
                  </div>
                </div>
              ))}
              <p style={{ fontSize: '.8rem', color: 'var(--slate)', marginTop: 10 }}>Classes balanced via augmentation to prevent training bias toward majority class.</p>
            </div>

            {/* ── MODEL COMPARISON SECTION ── */}
            <p className="section-title">Model Comparison</p>
            <p className="section-sub">CNN-only baseline vs. the hybrid CNN + CLIP ensemble across key performance indicators.</p>

            {/* Comparison table */}
            <div style={{ background: 'var(--white)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', overflow: 'hidden', boxShadow: 'var(--shadow-md)', marginBottom: 28 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '.9rem' }}>
                <thead>
                  <tr style={{ background: 'var(--navy)' }}>
                    {['Metric', 'CNN Only', 'CNN + CLIP ✦'].map((h, i) => (
                      <th key={h} style={{ padding: '14px 20px', textAlign: i === 0 ? 'left' : 'center', color: i === 2 ? 'var(--teal-lt)' : 'var(--white)', fontWeight: 700, fontSize: '.82rem', letterSpacing: '.05em', textTransform: 'uppercase' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Overall Accuracy',    '79%',  '83–90%',  false],
                    ['Melanoma Precision',  '81%',  '90%',     false],
                    ['Melanoma Recall',     '78%',  '88%',     false],
                    ['F1 Score',            '79%',  '89%',     false],
                    ['Explainability',      '❌ None','✅ Grad-CAM + CLIP', false],
                    ['Validation Method',   'Softmax only','Softmax + CLIP prompts', false],
                    ['Stage Estimation',    '❌','✅ Via CLIP prompts', false],
                  ].map(([metric, cnn, hybrid], i) => (
                    <tr key={metric} style={{ background: i % 2 === 0 ? 'var(--offwhite)' : 'var(--white)', borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '13px 20px', fontWeight: 600, color: 'var(--navy)' }}>{metric}</td>
                      <td style={{ padding: '13px 20px', textAlign: 'center', color: 'var(--slate)', fontFamily: 'JetBrains Mono, monospace', fontSize: '.88rem' }}>{cnn}</td>
                      <td style={{ padding: '13px 20px', textAlign: 'center', fontFamily: 'JetBrains Mono, monospace', fontSize: '.88rem', fontWeight: 700, color: 'var(--teal)', background: 'rgba(13,148,136,.05)' }}>{hybrid}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Bar chart comparison */}
            <div style={{ background: 'var(--white)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', padding: '28px', boxShadow: 'var(--shadow-sm)', marginBottom: 28 }}>
              <div style={{ fontSize: '.8rem', fontWeight: 700, letterSpacing: '.06em', textTransform: 'uppercase', color: 'var(--slate)', marginBottom: 20 }}>Accuracy Comparison — CNN vs CNN + CLIP</div>
              {[
                ['Overall Accuracy',   79, 87],
                ['Precision',          81, 90],
                ['Recall',             78, 88],
                ['F1 Score',           79, 89],
              ].map(([label, cnn, hybrid]) => (
                <div key={label} style={{ marginBottom: 18 }}>
                  <div style={{ fontSize: '.83rem', fontWeight: 600, color: 'var(--navy)', marginBottom: 6 }}>{label}</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ fontSize: '.72rem', color: 'var(--slate)', width: 72, textAlign: 'right' }}>CNN only</span>
                      <div style={{ flex: 1, height: 8, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                        <div style={{ width: `${cnn}%`, height: '100%', background: '#94a3b8', borderRadius: 99 }} />
                      </div>
                      <span style={{ fontSize: '.78rem', fontFamily: 'JetBrains Mono, monospace', color: 'var(--slate)', width: 36 }}>{cnn}%</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ fontSize: '.72rem', color: 'var(--teal)', width: 72, textAlign: 'right', fontWeight: 600 }}>CNN+CLIP</span>
                      <div style={{ flex: 1, height: 8, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                        <div style={{ width: `${hybrid}%`, height: '100%', background: 'var(--teal)', borderRadius: 99 }} />
                      </div>
                      <span style={{ fontSize: '.78rem', fontFamily: 'JetBrains Mono, monospace', color: 'var(--teal)', fontWeight: 700, width: 36 }}>{hybrid}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* CLIP prompt scores */}
            <p className="section-title" style={{ marginTop: 8 }}>CLIP Prompt Similarity Scores</p>
            <p className="section-sub">Example output for a melanoma-positive sample — scores represent cosine similarity to each clinical prompt.</p>
            <div style={{ background: 'var(--white)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', padding: '28px', boxShadow: 'var(--shadow-sm)', marginBottom: 48 }}>
              {[
                ['Early stage melanoma',        0.73, '#ef4444'],
                ['Intermediate stage melanoma', 0.35, '#f59e0b'],
                ['Advanced stage melanoma',     0.12, '#f97316'],
                ['Benign skin lesion',          0.24, '#10b981'],
              ].map(([prompt, score, color]) => (
                <div key={prompt} style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '.85rem', marginBottom: 5 }}>
                    <span style={{ color: 'var(--navy)', fontWeight: 500 }}>{`"${prompt}"`}</span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color }}>{score.toFixed(2)}</span>
                  </div>
                  <div style={{ height: 9, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{ width: `${score * 100}%`, height: '100%', background: color, borderRadius: 99, transition: 'width .9s ease' }} />
                  </div>
                </div>
              ))}
              <p style={{ fontSize: '.78rem', color: 'var(--slate)', marginTop: 12 }}>Highest score determines stage estimation shown in the prediction report.</p>
            </div>

            {/* Confusion matrix */}
            <p className="section-title">Confusion Matrix</p>
            <p className="section-sub">Evaluated on the held-out test set — 685 total samples.</p>
            <div style={{ background: 'var(--white)', border: '1px solid var(--border)', borderRadius: 'var(--r-lg)', padding: '28px', boxShadow: 'var(--shadow-sm)', marginBottom: 48, overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', margin: '0 auto' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '10px 16px', color: 'var(--slate)', fontSize: '.78rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.05em' }}>Actual \ Predicted</th>
                    <th style={{ padding: '10px 24px', color: '#10b981', fontSize: '.82rem', fontWeight: 700 }}>Benign</th>
                    <th style={{ padding: '10px 24px', color: '#ef4444', fontSize: '.82rem', fontWeight: 700 }}>Melanoma</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Benign',   [[300,'#d1fae5','#065f46'],[30,'#fee2e2','#991b1b']]],
                    ['Melanoma', [[45,'#fee2e2','#991b1b'],[310,'#d1fae5','#065f46']]],
                  ].map(([rowLabel, cells]) => (
                    <tr key={rowLabel}>
                      <td style={{ padding: '10px 16px', fontWeight: 700, color: 'var(--navy)', fontSize: '.88rem' }}>{rowLabel}</td>
                      {cells.map(([val, bg, fg], i) => (
                        <td key={i} style={{ padding: '18px 24px', textAlign: 'center', background: bg, borderRadius: 8, fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.2rem', color: fg, margin: 4 }}>{val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ display: 'flex', gap: 20, justifyContent: 'center', marginTop: 16, fontSize: '.78rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><span style={{ width: 12, height: 12, background: '#d1fae5', borderRadius: 3, display: 'inline-block' }} /> Correct predictions</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}><span style={{ width: 12, height: 12, background: '#fee2e2', borderRadius: 3, display: 'inline-block' }} /> Misclassifications</span>
              </div>
            </div>

          </div>
        </div>
      )}

      {route === 'team' && (
        <div className="section-wrap page-enter" id="team">
          <p className="section-title">Contact Us — Team</p>
          <p className="section-sub">Project guide and team contacts</p>

          {/* Guide on first row */}
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center', marginBottom: 20 }}>
            <div className="team-guide-card">
              <div style={{ width: 96, height: 96, borderRadius: 999, background: 'var(--offwhite)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '2.2rem' }} aria-hidden>
                👩‍🏫
              </div>
              <div>
                <div style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--navy)' }}>Mrs. Boosi Shyamala</div>
                <div style={{ fontSize: '.92rem', color: 'var(--slate)', marginTop: 6 }}>Assistant Professor, Dept. of CSE</div>
                <div style={{ fontSize: '.85rem', color: 'var(--slate)', marginTop: 8 }}>Gitam School</div>
              </div>
            </div>
          </div>

          {/* Four members in a row (second row) */}
          <div style={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
            <div className="team-grid">
              {[
                ['Team Leader', 'Veeresh Hedderi' ],
                ['Team Associate', 'Kumar Guttal' ],
                ['Team Member', 'RamKrishna'],
                ['Team Member', 'Kushal C'],
              ].map(([role, name, initial]) => {
                const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '_')
                return (
                  <div key={name} className="team-member">
                    <div className="team-avatar">
                      <img src={`team/${slug}.jpg`} alt={name} onError={e => { e.currentTarget.style.display = 'none' }} />
                    </div>
                    <div style={{ fontWeight: 700, color: 'var(--navy)', textAlign: 'center' }}>{name}</div>
                    <div style={{ fontSize: '.82rem', color: 'var(--slate)', textAlign: 'center' }}>{role}</div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
            {/* ── FOOTER ── */}
      <footer className="footer">
        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', width: '100%', flexWrap: 'wrap', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span className="footer-logo">✦ DermoDetection</span>
            <span style={{ fontSize: '.92rem', color: 'var(--slate-lt)' }}>AI-Powered Skin Cancer Detection · Research Use Only</span>
          </div>

          <div style={{ flex: '1 1 520px', maxWidth: 920 }}>
            <h2 style={{ margin: '0 0 6px 0' }}>Skin Cancer Detection From Dermoscopic Images · Capstone Project</h2>
            <div style={{ fontSize: '.95rem', color: 'var(--slate)', lineHeight: 1.5 }}>
              <strong>Team Details:</strong>
              <div>Team Leader: Veeresh Hedderi</div>
              <div>Team Associate: Kumar Guttal</div>
              <div>Team Member: RamKrishna</div>
              <div>Team Member: Kushal C</div>
              <div>Guided by: Mrs. Boosi Shaymaloa — Assistant Professor, Dept. of CSE</div>
              <div>Gitam School</div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
            <span style={{ color: 'var(--slate-lt)' }}>© {new Date().getFullYear()} — Not for clinical use</span>
          </div>
        </div>
      </footer>
    </div>
  )
}