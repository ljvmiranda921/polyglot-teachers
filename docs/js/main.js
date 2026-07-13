/* Small page-wide behaviors. */
document.addEventListener('DOMContentLoaded', () => {
  // Give each findings subsection a slug id and a hover-revealed "#" anchor
  // so it can be linked to directly.
  document.querySelectorAll('#findings h3').forEach((h) => {
    if (!h.id) {
      h.id = h.textContent.trim().toLowerCase()
        .replace(/[^\w\s-]/g, '').replace(/\s+/g, '-');
    }
    const a = document.createElement('a');
    a.className = 'hanchor';
    a.href = '#' + h.id;
    a.textContent = '#';
    a.setAttribute('aria-label', 'Link to this section');
    h.appendChild(a);
  });

  // Copy-to-clipboard buttons. A button with data-copy-target="#id" copies
  // the target element's textContent and briefly shows a confirmation label.
  document.querySelectorAll('.copy-btn[data-copy-target]').forEach((btn) => {
    const originalText = btn.textContent;
    let resetTimer = null;
    btn.addEventListener('click', async () => {
      const target = document.querySelector(btn.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent.trim());
        btn.textContent = 'Copied to clipboard';
        btn.classList.add('is-copied');
      } catch (err) {
        btn.textContent = 'Copy failed';
      }
      clearTimeout(resetTimer);
      resetTimer = setTimeout(() => {
        btn.textContent = originalText;
        btn.classList.remove('is-copied');
      }, 2000);
    });
  });
});
