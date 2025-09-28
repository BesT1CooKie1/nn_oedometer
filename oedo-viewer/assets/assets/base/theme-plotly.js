// Greift Plotly global ab und injiziert Layout/Config, basierend auf deinen CSS-Variablen.
// Keine Änderungen an deinem bestehenden Plot-Code nötig.

(function () {
  if (!window.Plotly) {
    console.warn("[plotly-theme] Plotly nicht geladen – Theme-Hook inaktiv.");
    return;
  }

  function cssVar(name, el = document.documentElement) {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || undefined;
  }

  function buildTheme() {
    const r = document.documentElement;
    const brand        = cssVar('--brand', r)        || '#0f7b86';
    const brand600     = cssVar('--brand-600', r)    || '#0c6670';
    const bg           = cssVar('--bg', r)           || '#f7f8fa';
    const surface      = cssVar('--surface', r)      || '#ffffff';
    const card         = cssVar('--card', r)         || surface;
    const text         = cssVar('--text', r)         || '#0b1220';
    const muted        = cssVar('--muted', r)        || '#5a6472';
    const border       = cssVar('--border', r)       || '#d7dbe1';
    const borderStrong = cssVar('--border-strong', r)|| '#b9c0c9';

    const colorway = [
      brand, brand600, '#2f4b7c', '#665191', '#a05195', '#d45087',
      '#f95d6a', '#ff7c43', '#ffa600', '#43aa8b', '#577590'
    ];

    const layout = {
      paper_bgcolor: card,
      plot_bgcolor: surface,
      colorway,
      font: {
        family: 'system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, Cantarell, "Noto Sans", Arial, sans-serif',
        color: text,
        size: 13
      },
      margin: { l: 56, r: 24, t: 24, b: 48 },
      xaxis: {
        gridcolor: border,
        zerolinecolor: borderStrong,
        linecolor: borderStrong,
        tickcolor: borderStrong,
        title: { font: { color: muted } },
        tickfont: { color: text }
      },
      yaxis: {
        gridcolor: border,
        zerolinecolor: borderStrong,
        linecolor: borderStrong,
        tickcolor: borderStrong,
        title: { font: { color: muted } },
        tickfont: { color: text }
      },
      legend: {
        bgcolor: surface,
        bordercolor: border,
        borderwidth: 1,
        font: { color: text }
      },
      hoverlabel: {
        bgcolor: surface,
        bordercolor: border,
        font: { color: text }
      },
      newshape: { line: { color: brand } },
      dragmode: 'pan'
    };

    const config = {
      displaylogo: false,
      responsive: true,
      toImageButtonOptions: { format: 'png', filename: 'plot', height: 720, width: 1080, scale: 2 },
      modeBarButtonsToRemove: ['autoScale2d', 'lasso2d', 'select2d', 'zoom2d']
    };

    return { layout, config, colors: { brand, muted } };
  }

  const THEME = buildTheme();

  // Modebar-Farben anpassen (CSS-in-JS, weil Tokens live sind)
  const style = document.createElement('style');
  style.textContent = `
    .modebar-btn svg { fill: ${THEME.colors.muted} !important; }
    .modebar-btn:hover svg, .modebar-btn.active svg { fill: ${THEME.colors.brand} !important; }
  `;
  document.head.appendChild(style);

  // Hook: newPlot und react mergen automatisch das Theme hinein
  const _newPlot = Plotly.newPlot.bind(Plotly);
  const _react   = Plotly.react.bind(Plotly);

  function mergeLayout(user) { return Object.assign({}, THEME.layout, user || {}); }
  function mergeConfig(user) { return Object.assign({}, THEME.config, user || {}); }

  Plotly.newPlot = function (gd, data, layout, config) {
    return _newPlot(gd, data, mergeLayout(layout), mergeConfig(config));
  };
  Plotly.react = function (gd, data, layout, config) {
    return _react(gd, data, mergeLayout(layout), mergeConfig(config));
  };

  console.info("[plotly-theme] aktiv – Layout/Config werden automatisch harmonisiert.");
})();
