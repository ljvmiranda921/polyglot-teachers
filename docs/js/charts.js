/* Findings charts, rendered with Plotly and styled to match chia (ink-outlined
   marks, brand palette, no modebar). Data is copied verbatim from the paper's
   tables. */
(function () {
  if (typeof Plotly === 'undefined') return;

  const BLUE = '#254eff', INK = '#1a1a1a', MUTED = '#c9c2b6';
  const FONT = { family: 'Univers, Helvetica, Arial, sans-serif', color: INK, size: 13 };
  const CONFIG = { displayModeBar: false, responsive: true };

  /* ---- Finding 1: PG-Score by teacher model ---- */

  const sota = {
    models: ['Gemma 3 27B Inst.', 'Aya Expanse 32B', 'Gemma 3 12B Inst.', 'Command A', 'Gemma 3 4B Inst.', 'GPT 4o mini', 'IBM Granite 4.0', 'IBM Granite Micro', 'Llama 3.1 70B Inst.', 'Llama 3.1 8B Inst.'],
    Average: [0.726, 0.706, 0.595, 0.546, 0.469, 0.461, 0.312, 0.304, 0.14, -0.356],
    ar: [0.145, -0.058, -0.464, -1.36, -0.488, -1.117, -0.072, -0.282, -0.964, -1.693],
    cs: [0.36, 0.222, 0.327, 0.114, 0.33, 0.015, -0.031, 0.29, 0.109, -0.974],
    de: [1.655, 1.468, 1.756, 1.673, 1.644, 1.766, 1.0, 1.102, 1.195, 0.891],
    es: [1.358, 1.129, 1.228, 1.102, 0.929, 0.908, 0.734, 0.783, 0.688, 0.182],
    id: [0.214, 1.153, 0.151, 1.063, -0.105, 1.003, -0.079, -0.329, 0.182, 0.322],
    ja: [0.626, 0.32, 0.573, 0.683, 0.504, 0.189, 0.321, 0.264, -0.373, -0.863],
  };

  const mount = document.getElementById('chart-teachers');
  if (mount) {
    // top-3 of the shown metric get the brand blue, the rest go muted
    const colorsFor = (vals) => {
      const cutoff = [...vals].sort((a, b) => b - a)[2];
      return vals.map((v) => (v >= cutoff ? BLUE : MUTED));
    };

    const trace = (key) => ({
      type: 'bar',
      orientation: 'h',
      y: sota.models,
      x: sota[key],
      marker: { color: colorsFor(sota[key]), line: { color: INK, width: 1 } },
      hovertemplate: '%{y}<br>PG-Score: %{x:.3f}<extra></extra>',
    });

    const layout = {
      margin: { l: 132, r: 20, t: 8, b: 34 },
      height: 380,
      font: FONT,
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
      bargap: 0.32,
      xaxis: {
        zeroline: true, zerolinecolor: INK, zerolinewidth: 1.5,
        gridcolor: '#eee', range: [-2, 2], title: { text: 'PG-Score', font: { size: 12 } },
      },
      yaxis: { autorange: 'reversed', ticks: '' },
    };

    Plotly.newPlot(mount, [trace('Average')], layout, CONFIG);

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    document.querySelectorAll('#chart-teachers-chips .chart-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('#chart-teachers-chips .chart-chip')
          .forEach((c) => c.classList.toggle('is-active', c === chip));
        const key = chip.dataset.key;
        if (reduceMotion) {
          Plotly.react(mount, [trace(key)], layout, CONFIG);
          return;
        }
        // Tween the bar lengths (and recolor) instead of snapping.
        Plotly.animate(mount, { data: [trace(key)] }, {
          transition: { duration: 500, easing: 'cubic-in-out' },
          frame: { duration: 500, redraw: false },
        });
      });
    });
  }

  /* ---- Finding 2: does model scale predict PG-Score? ---- */

  const strengthMount = document.getElementById('chart-strength');
  if (strengthMount) {
    const FAMILY = {
      Google: { color: '#254eff', x: [27, 12, 4], y: [0.726, 0.595, 0.469], names: ['Gemma 3 27B', 'Gemma 3 12B', 'Gemma 3 4B'] },
      Cohere: { color: '#C96A2E', x: [32, 104], y: [0.706, 0.546], names: ['Aya Expanse 32B', 'Command A'] },
      IBM: { color: '#4DB78C', x: [3, 0.4], y: [0.312, 0.304], names: ['IBM Granite 4.0', 'IBM Granite Micro'] },
      Meta: { color: '#A368DF', x: [70, 8], y: [0.14, -0.356], names: ['Llama 3.1 70B', 'Llama 3.1 8B'] },
    };

    const dots = Object.keys(FAMILY).map((fam) => ({
      type: 'scatter', mode: 'markers', name: fam,
      x: FAMILY[fam].x, y: FAMILY[fam].y, text: FAMILY[fam].names,
      marker: { size: 13, color: FAMILY[fam].color, line: { color: INK, width: 1 } },
      hovertemplate: '%{text}<br>%{x}B params<br>PG-Score: %{y:.3f}<extra></extra>',
    }));

    // the flat reality (OLS fit of the points above) vs. the steep line you'd
    // expect if bigger simply meant better
    const actual = {
      type: 'scatter', mode: 'lines', name: 'actual fit',
      x: [0.32, 131], y: [0.222, 0.493],
      line: { color: INK, width: 2 }, hoverinfo: 'skip',
    };
    const expected = {
      type: 'scatter', mode: 'lines', name: 'if scale mattered',
      x: [0.32, 131], y: [0.0, 1.0],
      line: { color: MUTED, width: 2, dash: 'dot' }, hoverinfo: 'skip',
    };

    const strengthLayout = {
      margin: { l: 48, r: 16, t: 8, b: 46 },
      height: 400,
      font: FONT,
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      legend: { orientation: 'h', y: -0.18, font: { size: 12 } },
      xaxis: {
        type: 'log', gridcolor: '#eee',
        tickvals: [1, 10, 100], ticktext: ['1B', '10B', '100B'],
        title: { text: 'Parameter size (log scale)', font: { size: 12 } },
      },
      yaxis: {
        zeroline: true, zerolinecolor: INK, zerolinewidth: 1.5, gridcolor: '#eee',
        title: { text: 'PG-Score', font: { size: 12 } },
      },
    };

    Plotly.newPlot(strengthMount, [expected, actual, ...dots], strengthLayout, CONFIG);
  }

  /* ---- Finding: which intrinsic metrics predict downstream performance? ----
     Reproduced by running analysis.principal_components on the committed
     data/csd3 + pg_scores_base_olmo3.jsonl (results_key=pg_score). */

  const pcaMount = document.getElementById('chart-pca-loadings');
  if (pcaMount) {
    const features = ['Distinct Prompts', 'Distinct Responses', 'Perplexity', 'Rubric Score', 'Prompt Length', 'Response Length'];
    const pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'];
    const loadings = [
      [-0.351, -0.018, -0.408, 0.688, 0.461, -0.159],
      [0.334, -0.482, 0.294, 0.555, -0.429, -0.280],
      [-0.361, 0.027, 0.773, 0.217, 0.229, 0.416],
      [0.566, 0.305, -0.192, 0.346, -0.017, 0.656],
      [-0.254, 0.738, 0.074, 0.230, -0.525, -0.237],
      [0.501, 0.358, 0.329, -0.004, 0.525, -0.487],
    ];
    const scale = [[0, '#C96A2E'], [0.5, '#f4f2ee'], [1, '#254eff']];
    const annotations = [];
    features.forEach((f, r) => pcs.forEach((p, c) => {
      const v = loadings[r][c];
      annotations.push({ x: p, y: f, text: v.toFixed(2), showarrow: false,
        font: { size: 10, color: Math.abs(v) > 0.5 ? '#fff' : INK } });
    }));

    Plotly.newPlot(pcaMount, [{
      type: 'heatmap', x: pcs, y: features, z: loadings,
      colorscale: scale, zmid: 0, zmin: -0.8, zmax: 0.8, xgap: 2, ygap: 2,
      showscale: false,
      hovertemplate: '%{y} on %{x}<br>loading: %{z:.3f}<extra></extra>',
    }], {
      margin: { l: 92, r: 8, t: 6, b: 26 },
      height: 320, font: FONT, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      xaxis: { side: 'bottom', tickfont: { size: 11 } },
      yaxis: { autorange: 'reversed', tickfont: { size: 10.5 } },
      annotations,
    }, CONFIG);
  }

  const pcaScatter = document.getElementById('chart-pca-scatter');
  if (pcaScatter) {
    const LANG = { de: ['German', '#254eff'], es: ['Spanish', '#A368DF'], id: ['Indonesian', '#C96A2E'], cs: ['Czech', '#4DB78C'], ja: ['Japanese', '#7B93B8'] };
    const pts = [{"x":0.264,"y":0.586,"l":"ja"},{"x":1.0,"y":0.656,"l":"de"},{"x":0.908,"y":0.969,"l":"es"},{"x":1.129,"y":1.035,"l":"es"},{"x":0.182,"y":0.507,"l":"id"},{"x":1.195,"y":0.701,"l":"de"},{"x":0.182,"y":0.575,"l":"es"},{"x":-0.329,"y":0.26,"l":"id"},{"x":0.214,"y":0.165,"l":"id"},{"x":0.322,"y":0.146,"l":"id"},{"x":0.151,"y":0.096,"l":"id"},{"x":1.153,"y":0.784,"l":"id"},{"x":1.102,"y":1.172,"l":"es"},{"x":1.003,"y":0.99,"l":"id"},{"x":-0.079,"y":-0.525,"l":"id"},{"x":1.102,"y":0.968,"l":"de"},{"x":0.688,"y":0.762,"l":"es"},{"x":0.929,"y":0.841,"l":"es"},{"x":0.109,"y":0.403,"l":"cs"},{"x":0.114,"y":0.358,"l":"cs"},{"x":0.015,"y":0.075,"l":"cs"},{"x":-0.105,"y":0.211,"l":"id"},{"x":0.783,"y":0.947,"l":"es"},{"x":0.321,"y":-0.073,"l":"ja"},{"x":0.734,"y":0.654,"l":"es"},{"x":-0.031,"y":0.046,"l":"cs"},{"x":1.468,"y":1.029,"l":"de"},{"x":0.29,"y":0.475,"l":"cs"}];
    const diag = { type: 'scatter', mode: 'lines', x: [-0.6, 1.6], y: [-0.6, 1.6], line: { color: MUTED, width: 1.5, dash: 'dash' }, hoverinfo: 'skip', showlegend: false };
    const traces = Object.keys(LANG).map((code) => {
      const p = pts.filter((d) => d.l === code);
      return { type: 'scatter', mode: 'markers', name: LANG[code][0],
        x: p.map((d) => d.x), y: p.map((d) => d.y),
        marker: { size: 9, color: LANG[code][1], line: { color: INK, width: 1 } },
        hovertemplate: LANG[code][0] + '<br>actual: %{x:.2f}<br>predicted: %{y:.2f}<extra></extra>' };
    });
    Plotly.newPlot(pcaScatter, [diag, ...traces], {
      margin: { l: 44, r: 8, t: 6, b: 40 }, height: 320, font: FONT,
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      legend: { orientation: 'h', y: -0.2, font: { size: 11 } },
      xaxis: { title: { text: 'Actual', font: { size: 12 } }, gridcolor: '#eee', zeroline: false },
      yaxis: { title: { text: 'Predicted', font: { size: 12 } }, gridcolor: '#eee', zeroline: false, scaleanchor: 'x', scaleratio: 1 },
    }, CONFIG);
  }
})();
