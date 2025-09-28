// modules/navbar/navbar.js
async function loadConfig() {
  // Pfad relativ zu diesem JS: /modules/navbar/nav.config.json
  const base = new URL(import.meta.url);
  const url  = new URL('nav.config.json', base).toString();
  const res  = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Navbar config load failed: ${res.status}`);
  return res.json();
}

function isActive(href) {
  try {
    const here = new URL(window.location.href);
    const link = new URL(href, here);
    // Vergleiche Pfad ohne Query/Hash
    return here.pathname.replace(/\/+$/, '') === link.pathname.replace(/\/+$/, '');
  } catch {
    return false;
  }
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === 'class') node.className = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function buildDropdown(label, items = []) {
  const btn = el('button', { type: 'button' }, label);
  const menu = el('div', { class: 'dropdown-menu', role: 'menu' },
    items.map(it => el('a', {
      href: it.href,
      target: it.target || (it.external ? '_blank' : null),
      rel: it.external ? 'noopener noreferrer' : null,
      role: 'menuitem'
    }, it.label))
  );
  const wrap = el('div', { class: 'dropdown', 'aria-expanded': 'false' }, btn, menu);

  btn.addEventListener('click', () => {
    const open = wrap.getAttribute('aria-expanded') === 'true';
    document.querySelectorAll('.navbar .dropdown').forEach(d => d.setAttribute('aria-expanded','false'));
    wrap.setAttribute('aria-expanded', String(!open));
  });

  document.addEventListener('click', (e) => {
    if (!wrap.contains(e.target)) wrap.setAttribute('aria-expanded', 'false');
  });

  return wrap;
}

function buildNavbar(cfg) {
  const header = document.getElementById('app-header');
  if (!header) return;

  const navbar = el('nav', { class: 'navbar', role: 'navigation', 'aria-expanded': 'false' });

  // Brand
  const brandContent = [];
  if (cfg.brand?.logo) brandContent.push(el('img', { src: cfg.brand.logo, alt: '' }));
  brandContent.push(cfg.brand?.title || 'App');
  const brand = el('a', { class: 'brand', href: cfg.brand?.href || '/' }, ...brandContent);

  // Burger
  const burger = el('button', { class: 'burger btn-sidebar', type: 'button', 'aria-label': 'Menü' }, '☰');
  burger.addEventListener('click', () => {
    const open = navbar.getAttribute('aria-expanded') === 'true';
    navbar.setAttribute('aria-expanded', String(!open));
  });

  // Items
  const items = el('div', { class: 'items' });
  (cfg.items || []).forEach(item => {
    if (item.children?.length) {
      items.appendChild(buildDropdown(item.label, item.children));
    } else {
      const a = el('a', { class: 'link', href: item.href }, item.label);
      if (isActive(item.href)) a.setAttribute('aria-current', 'page');
      if (item.external) {
        a.setAttribute('target', item.target || '_blank');
        a.setAttribute('rel', 'noopener noreferrer');
      }
      items.appendChild(a);
    }
  });

  // Spacer + (optional) Right-Slot
  const spacer = el('div', { class: 'spacer' });
  const right  = el('div', { class: 'right-slot' }); // für zukünftige Controls (Theme, User, etc.)

  navbar.append(brand, burger, items, spacer, right);
  header.replaceChildren(navbar);
}

(async () => {
  try {
    const cfg = await loadConfig();
    buildNavbar(cfg);
  } catch (err) {
    console.error('[navbar] init failed:', err);
  }
  })();
