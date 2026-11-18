// Ambient sci-fi effects: neural-constellation background, scanlines, boot sequence.
(function () {
  document.documentElement.setAttribute('data-theme', 'dark');
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var fx = !reduce;
  try { var s = localStorage.getItem('nnl-fx'); if (s !== null) fx = s === '1'; } catch (e) {}

  document.addEventListener('DOMContentLoaded', function () {
    document.body.classList.toggle('fx', fx);

    /* ---------- constellation background ---------- */
    var cv = document.createElement('canvas'); cv.id = 'fxcanvas'; document.body.prepend(cv);
    var ctx = cv.getContext('2d'), W = 0, H = 0, dpr = Math.min(2, window.devicePixelRatio || 1), P = [];
    function size() {
      W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr;
      cv.style.width = W + 'px'; cv.style.height = H + 'px'; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var n = Math.round(Math.min(90, W * H / 16000));
      P = [];
      for (var i = 0; i < n; i++) P.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .18, vy: (Math.random() - .5) * .18, r: Math.random() * 1.4 + .3, h: Math.random() < .18 });
    }
    size(); addEventListener('resize', size);
    var mx = -999, my = -999;
    addEventListener('mousemove', function (e) { mx = e.clientX; my = e.clientY; });
    var pulses = [];
    function frame() {
      ctx.clearRect(0, 0, W, H);
      if (document.body.classList.contains('fx')) {
        for (var i = 0; i < P.length; i++) {
          var p = P[i]; p.x += p.vx; p.y += p.vy;
          if (p.x < 0) p.x = W; if (p.x > W) p.x = 0; if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
        }
        ctx.lineWidth = 1;
        for (var a = 0; a < P.length; a++) for (var b = a + 1; b < P.length; b++) {
          var dx = P[a].x - P[b].x, dy = P[a].y - P[b].y, d = dx * dx + dy * dy;
          if (d < 15000) {
            ctx.strokeStyle = 'rgba(34,228,255,' + (0.11 * (1 - d / 15000)).toFixed(3) + ')';
            ctx.beginPath(); ctx.moveTo(P[a].x, P[a].y); ctx.lineTo(P[b].x, P[b].y); ctx.stroke();
            if (Math.random() < 0.0004 && pulses.length < 14) pulses.push({ a: P[a], b: P[b], t: 0 });
          }
        }
        for (var k = pulses.length - 1; k >= 0; k--) {
          var q = pulses[k]; q.t += 0.02;
          if (q.t >= 1) { pulses.splice(k, 1); continue; }
          var px = q.a.x + (q.b.x - q.a.x) * q.t, py = q.a.y + (q.b.y - q.a.y) * q.t;
          ctx.fillStyle = 'rgba(255,61,154,.9)'; ctx.shadowColor = '#ff3d9a'; ctx.shadowBlur = 8;
          ctx.beginPath(); ctx.arc(px, py, 1.8, 0, 7); ctx.fill(); ctx.shadowBlur = 0;
        }
        for (var j = 0; j < P.length; j++) {
          var s2 = P[j], near = Math.hypot(s2.x - mx, s2.y - my) < 120;
          ctx.fillStyle = s2.h ? 'rgba(167,123,255,.9)' : near ? 'rgba(34,228,255,1)' : 'rgba(180,230,255,.6)';
          ctx.beginPath(); ctx.arc(s2.x, s2.y, s2.r + (near ? 1 : 0), 0, 7); ctx.fill();
        }
      }
      if (!reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    /* ---------- FX toggle ---------- */
    var b = document.getElementById('fxbtn');
    function label() { if (b) b.textContent = 'FX ' + (document.body.classList.contains('fx') ? 'ON' : 'OFF'); }
    label();
    if (b) b.addEventListener('click', function () {
      var on = !document.body.classList.contains('fx');
      document.body.classList.toggle('fx', on); label();
      try { localStorage.setItem('nnl-fx', on ? '1' : '0'); } catch (e) {}
    });

    /* ---------- boot sequence (playground, once per session) ---------- */
    var bootEl = document.getElementById('boot');
    if (!bootEl) return;
    var seen = false; try { seen = sessionStorage.getItem('nnl-boot') === '1'; } catch (e) {}
    if (seen || reduce) { bootEl.remove(); return; }
    try { sessionStorage.setItem('nnl-boot', '1'); } catch (e) {}
    var lines = [
      ['> initializing neural core', 'OK'],
      ['> allocating synaptic weights', 'OK'],
      ['> calibrating gradient engine', 'OK'],
      ['> loading training datasets [8]', 'OK'],
      ['> backpropagation module', 'ONLINE'],
    ];
    var term = bootEl.querySelector('.term .lines'), bar = bootEl.querySelector('.bar2 i'), i2 = 0;
    function done() { bootEl.classList.add('hide'); setTimeout(function () { bootEl.remove(); }, 600); }
    bootEl.addEventListener('click', done);
    (function next() {
      if (!document.body.contains(bootEl)) return;
      if (i2 >= lines.length) { setTimeout(done, 350); return; }
      var d = document.createElement('div');
      d.innerHTML = lines[i2][0] + ' <span class="ok">[' + lines[i2][1] + ']</span>';
      term.appendChild(d); i2++; bar.style.width = (i2 / lines.length * 100) + '%';
      setTimeout(next, 220);
    })();
  });
})();
