/* Plays the three method cards (Generate / Translate / Respond) as chat
   conversations, card by card, once. Seed bubbles pop in; teacher bubbles
   show a typing indicator first. Starts when the cards scroll into view and
   honors prefers-reduced-motion by rendering the final state statically. */
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

  function play() {
    reset();
    var i = 0;
    (function next() {
      if (i >= items.length) return;  // played once, stay on the final state
      var el = items[i++];
      if (el.classList.contains('from-teacher')) {
        el.classList.add('is-typing');
        setTimeout(function () {
          el.classList.remove('is-typing');
          el.classList.add('is-shown');
          setTimeout(next, STEP_MS);
        }, TYPING_MS);
      } else {
        el.classList.add('is-shown');
        setTimeout(next, STEP_MS);
      }
    })();
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        io.disconnect();
        play();
      }
    });
  }, { threshold: 0.3 });
  io.observe(root);
})();
