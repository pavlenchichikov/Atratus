/* Signature FX for the G-Trade terminal: boot sequence, chart crosshair,
   regime-reactive ambient wash, and odometer count-up on hero numbers.
   Pure cosmetics, no dependencies, honours prefers-reduced-motion. */
(function () {
  "use strict";

  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- Boot sequence (plays once per browser session) ----
  function boot() {
    var el = document.getElementById("boot");
    if (!el) return;
    if (reduce || sessionStorage.getItem("gt_booted")) { el.remove(); return; }
    sessionStorage.setItem("gt_booted", "1");
    var lines = el.querySelectorAll(".boot-line");
    Array.prototype.forEach.call(lines, function (ln, i) {
      setTimeout(function () { ln.classList.add("show"); }, 140 + i * 175);
    });
    setTimeout(function () {
      el.classList.add("done");
      setTimeout(function () { if (el.parentNode) el.remove(); }, 600);
    }, 1320);
  }

  // ---- Chart crosshair that tracks the cursor (fine pointers only) ----
  function crosshair() {
    if (reduce) return;
    if (window.matchMedia && !window.matchMedia("(pointer: fine)").matches) return;
    var hx = document.querySelector(".xhair-x");
    var hy = document.querySelector(".xhair-y");
    if (!hx || !hy) return;
    var raf = null, mx = 0, my = 0;
    document.addEventListener("mousemove", function (e) {
      mx = e.clientX; my = e.clientY;
      hx.classList.add("on"); hy.classList.add("on");
      if (raf) return;
      raf = requestAnimationFrame(function () {
        hx.style.top = my + "px";
        hy.style.left = mx + "px";
        raf = null;
      });
    });
    document.addEventListener("mouseleave", function () {
      hx.classList.remove("on"); hy.classList.remove("on");
    });
  }

  // ---- Regime-reactive ambient: tint the floor by the bull/bear score ----
  function regimeTint() {
    fetch("/api/regime").then(function (r) { return r.json(); }).then(function (d) {
      var s = (d && typeof d.score === "number") ? d.score : 50;
      var tint;
      if (s >= 60) {
        tint = "rgba(0, 255, 163, " + (0.05 + (s - 60) / 40 * 0.06).toFixed(3) + ")";
      } else if (s <= 40) {
        tint = "rgba(255, 59, 92, " + (0.05 + (40 - s) / 40 * 0.06).toFixed(3) + ")";
      } else {
        tint = "rgba(120, 140, 170, 0.035)";
      }
      document.documentElement.style.setProperty("--regime-tint", tint);
    }).catch(function () {});
  }

  // ---- Odometer count-up for hero stat numbers ----
  function fmt(n, decimals, grouped) {
    var s = n.toFixed(decimals);
    if (grouped) {
      var p = s.split(".");
      p[0] = p[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
      s = p.join(".");
    }
    return s;
  }
  function countUp(node) {
    var raw = node.nodeValue;
    var m = raw.match(/^(\s*[-+]?\$?\s*)([\d,]*\.?\d+)(\s*%?\s*)$/);
    if (!m) return;
    var pre = m[1], core = m[2], post = m[3];
    var target = parseFloat(core.replace(/,/g, ""));
    if (!isFinite(target)) return;
    var decimals = (core.split(".")[1] || "").length;
    var grouped = core.indexOf(",") !== -1;
    var dur = 850, t0 = performance.now();
    function step(now) {
      var p = Math.min(1, (now - t0) / dur);
      var eased = 1 - Math.pow(1 - p, 3);
      node.nodeValue = pre + fmt(target * eased, decimals, grouped) + post;
      if (p < 1) requestAnimationFrame(step);
      else node.nodeValue = raw;
    }
    requestAnimationFrame(step);
  }
  function odometers() {
    if (reduce) return;
    document.querySelectorAll(".stat-value").forEach(function (el) {
      var tn = el.firstChild;
      if (tn && tn.nodeType === 3 && /\d/.test(tn.nodeValue)) countUp(tn);
    });
  }

  function run() { boot(); crosshair(); regimeTint(); odometers(); }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
