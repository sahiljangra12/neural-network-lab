(function () {
  var t = null;
  try { t = localStorage.getItem('nnl-theme'); } catch (e) {}
  if (!t) t = window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', t);
  document.addEventListener('DOMContentLoaded', function () {
    var b = document.getElementById('themebtn');
    if (!b) return;
    b.addEventListener('click', function () {
      var n = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', n);
      try { localStorage.setItem('nnl-theme', n); } catch (e) {}
      window.dispatchEvent(new Event('themechange'));
    });
  });
})();
