/* Small page-wide behaviors. */
document.addEventListener('DOMContentLoaded', () => {
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
