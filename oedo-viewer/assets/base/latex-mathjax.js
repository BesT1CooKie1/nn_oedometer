// js/latex-mathjax.js
(function () {
  function typeset(target) {
    // MathJax v3 benötigt typesetPromise für dynamische Inhalte
    if (window.MathJax && MathJax.typesetPromise) {
      const nodes = target ? [target] : undefined;
      return MathJax.typesetPromise(nodes).catch(() => {});
    }
    return Promise.resolve();
  }

  // Debounce, falls du sehr häufig updatest
  let t = null;
  function latexTypeset(target) {
    if (t) cancelAnimationFrame(t);
    t = requestAnimationFrame(() => typeset(target));
  }

  // global verfügbar machen
  window.latexTypeset = latexTypeset;
})();
