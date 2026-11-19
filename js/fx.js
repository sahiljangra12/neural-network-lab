// Quiet starfield background and a short launch sequence.
(function () {
  document.documentElement.setAttribute('data-theme', 'dark');
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var fx = true;
  try { var s = localStorage.getItem('nnl-fx'); if (s !== null) fx = s === '1'; } catch (e) {}

  document.addEventListener('DOMContentLoaded', function () {
    document.body.classList.toggle('fx', fx);

    /* ---------- starfield ---------- */
    var cv = document.createElement('canvas'); cv.id = 'fxcanvas'; document.body.prepend(cv);
    var ctx = cv.getContext('2d'), W = 0, H = 0, dpr = Math.min(2, window.devicePixelRatio || 1), S = [];
    function size() {
      W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr;
      cv.style.width = W + 'px'; cv.style.height = H + 'px'; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      S = [];
      var n = Math.round(W * H / 5000);
      for (var i = 0; i < n; i++) S.push({ x: Math.random() * W, y: Math.random() * H, r: Math.random() < .92 ? Math.random() * .7 + .2 : Math.random() * .6 + .9, a: Math.random() * .6 + .15, tw: Math.random() * 6.28, sp: Math.random() * .02 + .004 });
    }
    size(); addEventListener('resize', size);
    var sat = null, t = 0;
    function frame() {
      t++;
      ctx.clearRect(0, 0, W, H);
      if (document.body.classList.contains('fx')) {
        for (var i = 0; i < S.length; i++) {
          var p = S[i]; p.x -= p.sp; if (p.x < 0) p.x = W;
          var a = p.a * (0.75 + 0.25 * Math.sin(p.tw + t * 0.01));
          ctx.fillStyle = 'rgba(255,255,255,' + a.toFixed(3) + ')';
          ctx.fillRect(p.x, p.y, p.r, p.r);
        }
        // an occasional satellite crossing the sky
        if (!sat && Math.random() < 0.002) sat = { x: -10, y: Math.random() * H * .6, vx: .35 + Math.random() * .3, vy: (Math.random() - .5) * .12 };
        if (sat) {
          sat.x += sat.vx; sat.y += sat.vy;
          ctx.fillStyle = 'rgba(255,255,255,.7)'; ctx.fillRect(sat.x, sat.y, 1.4, 1.4);
          if (sat.x > W + 10) sat = null;
        }
      }
      if (!reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    /* ---------- toggle ---------- */
    var b = document.getElementById('fxbtn');
    function label() { if (b) b.textContent = 'STARS ' + (document.body.classList.contains('fx') ? 'ON' : 'OFF'); }
    label();
    if (b) b.addEventListener('click', function () {
      var on = !document.body.classList.contains('fx');
      document.body.classList.toggle('fx', on); label();
      try { localStorage.setItem('nnl-fx', on ? '1' : '0'); } catch (e) {}
    });

    /* ---------- launch sequence (once per session) ---------- */
    var bootEl = document.getElementById('boot');
    if (!bootEl) return;
    var seen = false; try { seen = sessionStorage.getItem('nnl-boot') === '1'; } catch (e) {}
    if (seen || reduce) { bootEl.remove(); return; }
    try { sessionStorage.setItem('nnl-boot', '1'); } catch (e) {}
    var lines = [
      ['T−00:04', 'NETWORK INITIALIZATION', 'GO'],
      ['T−00:03', 'WEIGHTS RANDOMIZED', 'GO'],
      ['T−00:02', 'GRADIENT ENGINE', 'GO'],
      ['T−00:01', 'TRAINING DATA LOADED', 'GO'],
    ];
    var term = bootEl.querySelector('.term .lines'), bar = bootEl.querySelector('.bar2 i'), i2 = 0;
    function done() { bootEl.classList.add('hide'); setTimeout(function () { bootEl.remove(); }, 650); }
    bootEl.addEventListener('click', done);
    (function next() {
      if (!document.body.contains(bootEl)) return;
      if (i2 >= lines.length) { setTimeout(done, 380); return; }
      var d = document.createElement('div');
      d.innerHTML = lines[i2][0] + '&nbsp;&nbsp;' + lines[i2][1] + ' <span class="ok" style="float:right">' + lines[i2][2] + '</span>';
      term.appendChild(d); i2++; bar.style.width = (i2 / lines.length * 100) + '%';
      setTimeout(next, 260);
    })();
  });
})();
