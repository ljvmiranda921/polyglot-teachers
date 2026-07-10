/* Clickable metric chips in the intrinsic/extrinsic boxes: clicking a chip
   shows its description and computation/benchmark. One active item per box. */
(function () {
  document.querySelectorAll('.metric-box').forEach(function (box) {
    var btns = box.querySelectorAll('.metric-btn');
    var items = box.querySelectorAll('.metric-detail-item');
    btns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        btns.forEach(function (b) { b.classList.toggle('is-active', b === btn); });
        items.forEach(function (it) {
          it.classList.toggle('is-active', it.dataset.metric === btn.dataset.metric);
        });
      });
    });
  });
})();
