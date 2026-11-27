// Ambient effects for the futuristic light theme: drifting node network, scroll reveal, boot sequence.
(function () {
  var theme = 'dark';
  try { var st = localStorage.getItem('nnl-theme'); if (st === 'light' || st === 'dark') theme = st; } catch (e) {}
  document.documentElement.setAttribute('data-theme', theme);
  var dark = function () { return document.documentElement.getAttribute('data-theme') !== 'light'; };
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  document.addEventListener('DOMContentLoaded', function () {
    /* ---------- background: slow network of nodes with red signal pulses ---------- */
    if (!reduce) {
      var cv = document.createElement('canvas'); cv.id = 'fxcanvas'; document.body.prepend(cv);
      var ctx = cv.getContext('2d'), W = 0, H = 0, dpr = Math.min(2, window.devicePixelRatio || 1), P = [], pulses = [], mx = -999, my = -999;
      var size = function () {
        W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        var n = Math.round(Math.min(70, W * H / 22000)); P = [];
        for (var i = 0; i < n; i++) P.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .22, vy: (Math.random() - .5) * .22, r: Math.random() * 1.4 + .8 });
      };
      size(); addEventListener('resize', size);
      addEventListener('mousemove', function (e) { mx = e.clientX; my = e.clientY; }, { passive: true });
      var frame = function () {
        if (!document.hidden) {
          ctx.clearRect(0, 0, W, H);
          for (var i = 0; i < P.length; i++) { var p = P[i]; p.x += p.vx; p.y += p.vy; if (p.x < 0) p.x = W; if (p.x > W) p.x = 0; if (p.y < 0) p.y = H; if (p.y > H) p.y = 0; }
          ctx.lineWidth = 1;
          for (var a = 0; a < P.length; a++) for (var b = a + 1; b < P.length; b++) {
            var dx = P[a].x - P[b].x, dy = P[a].y - P[b].y, d = dx * dx + dy * dy;
            if (d < 26000) {
              ctx.strokeStyle = (dark() ? 'rgba(235,240,245,' : 'rgba(11,11,11,') + ((dark() ? 0.16 : 0.13) * (1 - d / 26000)).toFixed(3) + ')';
              ctx.beginPath(); ctx.moveTo(P[a].x, P[a].y); ctx.lineTo(P[b].x, P[b].y); ctx.stroke();
              if (pulses.length < 10 && Math.random() < 0.0006) pulses.push({ a: P[a], b: P[b], t: 0 });
            }
          }
          for (var k = pulses.length - 1; k >= 0; k--) {
            var q = pulses[k]; q.t += 0.018; if (q.t >= 1) { pulses.splice(k, 1); continue; }
            ctx.fillStyle = dark() ? 'rgba(255,64,46,.95)' : 'rgba(208,23,10,.9)'; ctx.beginPath(); ctx.arc(q.a.x + (q.b.x - q.a.x) * q.t, q.a.y + (q.b.y - q.a.y) * q.t, 2.4, 0, 7); ctx.fill();
          }
          for (var j = 0; j < P.length; j++) {
            var s = P[j], near = Math.hypot(s.x - mx, s.y - my) < 130;
            ctx.fillStyle = near ? (dark() ? 'rgba(255,64,46,.95)' : 'rgba(208,23,10,.9)') : (dark() ? 'rgba(235,240,245,.4)' : 'rgba(11,11,11,.32)'); ctx.beginPath(); ctx.arc(s.x, s.y, s.r + (near ? 1.2 : 0), 0, 7); ctx.fill();
          }
        }
        requestAnimationFrame(frame);
      };
      requestAnimationFrame(frame);
    }

    /* ---------- theme switch (dark by default) ---------- */
    var tb = document.getElementById('themebtn');
    var paint = function () {
      if (tb) { tb.textContent = dark() ? 'LIGHT MODE' : 'DARK MODE'; tb.setAttribute('aria-label', 'Switch to ' + (dark() ? 'light' : 'dark') + ' mode'); }
      document.querySelectorAll('.posname').forEach(function (el) { el.textContent = dark() ? 'white' : 'black'; });
      var m = document.querySelector('meta[name=theme-color]'); if (m) m.content = dark() ? '#07090c' : '#eef1f4';
    };
    paint();
    if (tb) tb.addEventListener('click', function () {
      var n = dark() ? 'light' : 'dark'; document.documentElement.setAttribute('data-theme', n);
      try { localStorage.setItem('nnl-theme', n); } catch (e) {}
      paint(); window.dispatchEvent(new Event('themechange'));
    });

    /* ---------- scroll reveal ---------- */
    var els = document.querySelectorAll('.section, .guide .card, .cards .card, .lesson .widget, .lesson .callout, .reportbox, .inspo, .steps3 a');
    if (reduce || !('IntersectionObserver' in window)) return;
    var io = new IntersectionObserver(function (en) { en.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }); }, { rootMargin: '0px 0px -40px 0px', threshold: 0.05 });
    els.forEach(function (el, i) { el.classList.add('rv'); if (el.matches('.card, .steps3 a')) el.style.transitionDelay = (i % 4) * 70 + 'ms'; io.observe(el); });
    // safety: never leave content hidden
    setTimeout(function () { document.querySelectorAll('.rv:not(.in)').forEach(function (el) { var r = el.getBoundingClientRect(); if (r.top < innerHeight) el.classList.add('in'); }); }, 1200);

    /* ---------- boot sequence: console page, once per session ---------- */
    if (!document.querySelector('.app')) return;
    var seen = false; try { seen = sessionStorage.getItem('nnl-boot') === '1'; } catch (e) {}
    if (seen) return;
    try { sessionStorage.setItem('nnl-boot', '1'); } catch (e) {}
    var boot = document.createElement('div'); boot.id = 'boot';
    boot.innerHTML = '<div class="term"><div class="logo-big">NEURAL NETWORK LAB</div><div class="lines"></div><div class="bar2"><i></i></div><div class="skip">Click to skip</div></div>';
    document.body.appendChild(boot);
    var lines = ['INITIALIZING NETWORK', 'RANDOMIZING WEIGHTS', 'GRADIENT ENGINE', 'TRAINING DATA'], i2 = 0, term = boot.querySelector('.lines'), bar = boot.querySelector('.bar2 i');
    var done = function () { boot.classList.add('hide'); setTimeout(function () { boot.remove(); }, 500); };
    boot.addEventListener('click', done);
    (function next() {
      if (!document.body.contains(boot)) return;
      if (i2 >= lines.length) { setTimeout(done, 260); return; }
      var d = document.createElement('div'); d.innerHTML = '&gt; ' + lines[i2] + ' <span class="ok">READY</span>'; term.appendChild(d);
      i2++; bar.style.width = (i2 / lines.length * 100) + '%'; setTimeout(next, 190);
    })();
  });
})();
