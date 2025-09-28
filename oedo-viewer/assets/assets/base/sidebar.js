// === Sidebar mobile toggle ===============================================
function installSidebarToggle() {
  const btn = document.getElementById('btn-sidebar');
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebar-backdrop');

  if (!btn || !sidebar) return;

  function open() {
    document.body.classList.add('sidebar-open');
    btn.setAttribute('aria-expanded', 'true');
    if (backdrop) backdrop.removeAttribute('hidden');
  }
  function close() {
    document.body.classList.remove('sidebar-open');
    btn.setAttribute('aria-expanded', 'false');
    if (backdrop) backdrop.setAttribute('hidden', '');
  }
  function toggle() {
    if (document.body.classList.contains('sidebar-open')) close(); else open();
  }

  // Click handlers
  btn.addEventListener('click', toggle);
  backdrop?.addEventListener('click', close);

  // Escape key closes
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') close();
  });

  // On resize to desktop, ensure it’s open in normal layout (or just remove the class)
  const BREAKPOINT = 900;
  let wasMobile = window.innerWidth <= BREAKPOINT;
  window.addEventListener('resize', () => {
    const isMobile = window.innerWidth <= BREAKPOINT;
    if (!isMobile && wasMobile) {
      // leaving mobile → ensure clean state
      close();
    }
    wasMobile = isMobile;
  });
}

// Last existing line: window.addEventListener("DOMContentLoaded", init);
// Replace with:
window.addEventListener("DOMContentLoaded", () => {
  init();                  // your existing boot
  installSidebarToggle();  // add this line
});
