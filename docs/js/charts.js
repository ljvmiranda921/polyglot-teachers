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
     Faithful to the paper's figures: loadings read from the pca_loading_factors
     PDF text layer; scatter points recovered from the pca_predicted_vs_actual
     PDF vector data (58 teacher-language points across all six languages). */

  const pcaMount = document.getElementById('chart-pca-loadings');
  if (pcaMount) {
    const features = ['Distinct Prompts', 'Distinct Responses', 'Perplexity', 'Rubric Score', 'Prompt Length', 'Response Length'];
    const pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'];
    const loadings = [
      [0.073, 0.654, 0.008, 0.744, 0.012, -0.117],
      [0.579, -0.098, -0.017, 0.111, -0.660, 0.456],
      [-0.578, -0.037, 0.017, 0.211, 0.075, 0.784],
      [0.514, -0.237, 0.354, 0.182, 0.678, 0.247],
      [-0.079, 0.388, 0.838, -0.332, -0.171, 0.048],
      [-0.234, -0.596, 0.415, 0.497, -0.265, -0.318],
    ];
    const scale = [[0, '#C96A2E'], [0.5, '#f4f2ee'], [1, '#254eff']];
    const annotations = [];
    features.forEach((f, r) => pcs.forEach((p, c) => {
      const v = loadings[r][c];
      annotations.push({ x: p, y: f, text: v.toFixed(3), showarrow: false,
        font: { size: 9, color: Math.abs(v) > 0.5 ? '#fff' : INK } });
    }));

    Plotly.newPlot(pcaMount, [{
      type: 'heatmap', x: pcs, y: features, z: loadings,
      colorscale: scale, zmid: 0, zmin: -0.85, zmax: 0.85, xgap: 2, ygap: 2,
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
    // paper marker colors, keyed by language
    const LANG = { Arabic: '#7fd8c9', Czech: '#fd8254', German: '#cdb4e8', Spanish: '#29337a', Indonesian: '#ffb81c', Japanese: '#545e73' };
    const pts = [{"x":0.3411,"y":0.4041,"l":"Arabic"},{"x":0.3572,"y":0.4364,"l":"Arabic"},{"x":0.3518,"y":0.437,"l":"Arabic"},{"x":0.3174,"y":0.4092,"l":"Arabic"},{"x":0.3459,"y":0.4105,"l":"Arabic"},{"x":0.3515,"y":0.3783,"l":"Arabic"},{"x":0.3405,"y":0.4146,"l":"Arabic"},{"x":0.3279,"y":0.4121,"l":"Arabic"},{"x":0.3087,"y":0.3421,"l":"Arabic"},{"x":0.3592,"y":0.4245,"l":"Czech"},{"x":0.362,"y":0.4192,"l":"Czech"},{"x":0.3629,"y":0.4281,"l":"Czech"},{"x":0.3621,"y":0.4279,"l":"Czech"},{"x":0.3563,"y":0.4253,"l":"Czech"},{"x":0.3564,"y":0.4093,"l":"Czech"},{"x":0.3538,"y":0.3968,"l":"Czech"},{"x":0.3525,"y":0.3968,"l":"Czech"},{"x":0.361,"y":0.4268,"l":"Czech"},{"x":0.3276,"y":0.3003,"l":"Czech"},{"x":0.3798,"y":0.4397,"l":"German"},{"x":0.3849,"y":0.4444,"l":"German"},{"x":0.3769,"y":0.4311,"l":"German"},{"x":0.3997,"y":0.4602,"l":"German"},{"x":0.3971,"y":0.4596,"l":"German"},{"x":0.3968,"y":0.4641,"l":"German"},{"x":0.3825,"y":0.4563,"l":"German"},{"x":0.4,"y":0.4754,"l":"German"},{"x":0.3975,"y":0.4794,"l":"German"},{"x":0.3921,"y":0.4626,"l":"German"},{"x":0.3773,"y":0.4563,"l":"Spanish"},{"x":0.3832,"y":0.4621,"l":"Spanish"},{"x":0.3582,"y":0.4528,"l":"Spanish"},{"x":0.3825,"y":0.4626,"l":"Spanish"},{"x":0.3715,"y":0.4513,"l":"Spanish"},{"x":0.3779,"y":0.4573,"l":"Spanish"},{"x":0.374,"y":0.4541,"l":"Spanish"},{"x":0.3727,"y":0.443,"l":"Spanish"},{"x":0.3892,"y":0.4789,"l":"Spanish"},{"x":0.3858,"y":0.4647,"l":"Spanish"},{"x":0.3582,"y":0.4288,"l":"Indonesian"},{"x":0.3814,"y":0.4519,"l":"Indonesian"},{"x":0.3447,"y":0.4166,"l":"Indonesian"},{"x":0.359,"y":0.4115,"l":"Indonesian"},{"x":0.3619,"y":0.4139,"l":"Indonesian"},{"x":0.3574,"y":0.4105,"l":"Indonesian"},{"x":0.3838,"y":0.4435,"l":"Indonesian"},{"x":0.3798,"y":0.4349,"l":"Indonesian"},{"x":0.3513,"y":0.3678,"l":"Indonesian"},{"x":0.3506,"y":0.4114,"l":"Indonesian"},{"x":0.3603,"y":0.4404,"l":"Japanese"},{"x":0.3618,"y":0.4485,"l":"Japanese"},{"x":0.3714,"y":0.4539,"l":"Japanese"},{"x":0.3685,"y":0.4482,"l":"Japanese"},{"x":0.3699,"y":0.4425,"l":"Japanese"},{"x":0.3584,"y":0.4446,"l":"Japanese"},{"x":0.3618,"y":0.3993,"l":"Japanese"},{"x":0.3306,"y":0.3708,"l":"Japanese"},{"x":0.3667,"y":0.4441,"l":"Japanese"}];
    const diag = { type: 'scatter', mode: 'lines', x: [0.30, 0.41], y: [0.30, 0.41], line: { color: MUTED, width: 1.5, dash: 'dash' }, hoverinfo: 'skip', showlegend: false };
    const traces = Object.keys(LANG).map((lang) => {
      const p = pts.filter((d) => d.l === lang);
      return { type: 'scatter', mode: 'markers', name: lang,
        x: p.map((d) => d.x), y: p.map((d) => d.y),
        marker: { size: 9, color: LANG[lang], line: { color: INK, width: 1 } },
        hovertemplate: lang + '<br>actual: %{x:.3f}<br>predicted: %{y:.3f}<extra></extra>' };
    });
    Plotly.newPlot(pcaScatter, [diag, ...traces], {
      margin: { l: 48, r: 10, t: 6, b: 40 }, height: 320, font: FONT,
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      legend: { orientation: 'h', y: -0.2, font: { size: 11 } },
      xaxis: { title: { text: 'Actual Benchmark Score', font: { size: 12 } }, gridcolor: '#eee', zeroline: false, range: [0.305, 0.405] },
      yaxis: { title: { text: 'Predicted', font: { size: 12 } }, gridcolor: '#eee', zeroline: false, range: [0.29, 0.49] },
      annotations: [{ xref: 'paper', yref: 'paper', x: 0.97, y: 0.06, text: 'R<sup>2</sup> = 0.664<br>RMSE = 0.440', showarrow: false, align: 'right', bordercolor: INK, borderwidth: 1, borderpad: 4, bgcolor: '#fff', font: { size: 11 } }],
    }, CONFIG);
  }
})();
