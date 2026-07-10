/* Plays the three method cards (Generate / Translate / Respond) as chat
   conversations, card by card, then loops after a hold. Seed bubbles pop in;
   teacher bubbles show a typing indicator first. Starts when the cards scroll
   into view, resets when they leave, and honors prefers-reduced-motion by
   rendering the final state statically. */
(function () {
  var root = document.getElementById('method-cards');
  if (!root) return;

  var items = [];
  root.querySelectorAll('.method-card').forEach(function (card) {
    card.querySelectorAll('.chat-item').forEach(function (el) { items.push(el); });
  });

  function showAll() {
    items.forEach(function (el) { el.classList.add('is-shown'); });
  }
  function reset() {
    items.forEach(function (el) { el.classList.remove('is-shown', 'is-typing'); });
  }

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    showAll();
    return;
  }

  var STEP_MS = 500;    // pause after a bubble lands
  var TYPING_MS = 750;  // how long the teacher "types"
  var HOLD_MS = 4500;   // hold the finished state before looping

  var timer = null;
  var running = false;

  function play() {
    reset();
    var i = 0;
    (function next() {
      if (i >= items.length) {
        timer = setTimeout(play, HOLD_MS);
        return;
      }
      var el = items[i++];
      if (el.classList.contains('from-teacher')) {
        el.classList.add('is-typing');
        timer = setTimeout(function () {
          el.classList.remove('is-typing');
          el.classList.add('is-shown');
          timer = setTimeout(next, STEP_MS);
        }, TYPING_MS);
      } else {
        el.classList.add('is-shown');
        timer = setTimeout(next, STEP_MS);
      }
    })();
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting && !running) {
        running = true;
        play();
      } else if (!entry.isIntersecting && running) {
        running = false;
        clearTimeout(timer);
        reset();
      }
    });
  }, { threshold: 0.3 });
  io.observe(root);
})();
