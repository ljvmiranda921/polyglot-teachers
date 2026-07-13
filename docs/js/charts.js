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
      margin: { l: 128, r: 16, t: 6, b: 32 },
      height: 344,
      font: FONT,
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
      bargap: 0.3,
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
    const features = ['Distinct<br>Prompts', 'Distinct<br>Responses', 'Perplexity', 'Rubric<br>Score', 'Prompt<br>Length', 'Response<br>Length'];
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
        font: { size: 11, color: Math.abs(v) > 0.5 ? '#fff' : INK } });
    }));

    Plotly.newPlot(pcaMount, [{
      type: 'heatmap', x: pcs, y: features, z: loadings,
      colorscale: scale, zmid: 0, zmin: -0.85, zmax: 0.85, xgap: 2, ygap: 2,
      showscale: false,
      hovertemplate: '%{y} on %{x}<br>loading: %{z:.3f}<extra></extra>',
    }], {
      margin: { l: 92, r: 8, t: 6, b: 26 },
      height: 320, font: FONT, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      xaxis: { side: 'bottom', tickfont: { size: 12 } },
      yaxis: { autorange: 'reversed', tickfont: { size: 12 } },
      annotations,
    }, CONFIG);
  }

  const pcaScatter = document.getElementById('chart-pca-scatter');
  if (pcaScatter) {
    // paper marker colors, keyed by language
    const LANG = { Arabic: '#7fd8c9', Czech: '#fd8254', German: '#cdb4e8', Spanish: '#29337a', Indonesian: '#ffb81c', Japanese: '#545e73' };
    const MARK = { Arabic: 'circle', Czech: 'square', German: 'diamond', Spanish: 'triangle-up', Indonesian: 'cross', Japanese: 'x' };
    const pts = [{"x":0.3411,"y":0.4038,"l":"Arabic"},{"x":0.3572,"y":0.4362,"l":"Arabic"},{"x":0.3518,"y":0.4368,"l":"Arabic"},{"x":0.3174,"y":0.4089,"l":"Arabic"},{"x":0.3459,"y":0.4103,"l":"Arabic"},{"x":0.3515,"y":0.3781,"l":"Arabic"},{"x":0.3405,"y":0.4144,"l":"Arabic"},{"x":0.3279,"y":0.4118,"l":"Arabic"},{"x":0.3086,"y":0.3419,"l":"Arabic"},{"x":0.3592,"y":0.4242,"l":"Czech"},{"x":0.362,"y":0.419,"l":"Czech"},{"x":0.3629,"y":0.4279,"l":"Czech"},{"x":0.3621,"y":0.4277,"l":"Czech"},{"x":0.3562,"y":0.425,"l":"Czech"},{"x":0.3564,"y":0.4091,"l":"Czech"},{"x":0.3538,"y":0.3965,"l":"Czech"},{"x":0.3525,"y":0.3966,"l":"Czech"},{"x":0.361,"y":0.4265,"l":"Czech"},{"x":0.3276,"y":0.3,"l":"Czech"},{"x":0.3798,"y":0.4394,"l":"German"},{"x":0.3849,"y":0.4442,"l":"German"},{"x":0.3769,"y":0.4308,"l":"German"},{"x":0.3997,"y":0.46,"l":"German"},{"x":0.3971,"y":0.4594,"l":"German"},{"x":0.3968,"y":0.4639,"l":"German"},{"x":0.3825,"y":0.4561,"l":"German"},{"x":0.4,"y":0.4752,"l":"German"},{"x":0.3975,"y":0.4792,"l":"German"},{"x":0.3921,"y":0.4623,"l":"German"},{"x":0.3773,"y":0.4561,"l":"Spanish"},{"x":0.3832,"y":0.4619,"l":"Spanish"},{"x":0.3582,"y":0.4525,"l":"Spanish"},{"x":0.3825,"y":0.4623,"l":"Spanish"},{"x":0.3715,"y":0.451,"l":"Spanish"},{"x":0.3779,"y":0.4571,"l":"Spanish"},{"x":0.374,"y":0.4539,"l":"Spanish"},{"x":0.3727,"y":0.4428,"l":"Spanish"},{"x":0.3892,"y":0.4787,"l":"Spanish"},{"x":0.3858,"y":0.4645,"l":"Spanish"},{"x":0.3582,"y":0.4285,"l":"Indonesian"},{"x":0.3814,"y":0.4516,"l":"Indonesian"},{"x":0.3447,"y":0.4164,"l":"Indonesian"},{"x":0.359,"y":0.4113,"l":"Indonesian"},{"x":0.3619,"y":0.4137,"l":"Indonesian"},{"x":0.3573,"y":0.4103,"l":"Indonesian"},{"x":0.3838,"y":0.4433,"l":"Indonesian"},{"x":0.3798,"y":0.4346,"l":"Indonesian"},{"x":0.3513,"y":0.3676,"l":"Indonesian"},{"x":0.3506,"y":0.4112,"l":"Indonesian"},{"x":0.3603,"y":0.4401,"l":"Japanese"},{"x":0.3618,"y":0.4483,"l":"Japanese"},{"x":0.3714,"y":0.4537,"l":"Japanese"},{"x":0.3685,"y":0.448,"l":"Japanese"},{"x":0.3699,"y":0.4423,"l":"Japanese"},{"x":0.3583,"y":0.4443,"l":"Japanese"},{"x":0.3618,"y":0.3991,"l":"Japanese"},{"x":0.3306,"y":0.3705,"l":"Japanese"},{"x":0.3667,"y":0.4439,"l":"Japanese"}];
    const diag = { type: 'scatter', mode: 'lines', x: [0.30, 0.40], y: [0.30, 0.50], line: { color: '#545e73', width: 2, dash: 'dash' }, hoverinfo: 'skip', showlegend: false };
    const traces = Object.keys(LANG).map((lang) => {
      const p = pts.filter((d) => d.l === lang);
      return { type: 'scatter', mode: 'markers', name: lang,
        x: p.map((d) => d.x), y: p.map((d) => d.y),
        marker: { size: 11, symbol: MARK[lang], color: LANG[lang], line: { color: INK, width: 1 } },
        hovertemplate: lang + '<br>actual: %{x:.3f}<br>predicted: %{y:.3f}<extra></extra>' };
    });
    Plotly.newPlot(pcaScatter, [diag, ...traces], {
      margin: { l: 48, r: 10, t: 6, b: 88 }, height: 380, font: FONT,
      paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      legend: { orientation: 'h', y: -0.2, yanchor: 'top', xanchor: 'center', x: 0.5, font: { size: 11 } },
      xaxis: { title: { text: 'Actual Benchmark Score', font: { size: 12 } }, gridcolor: '#eee', zeroline: false, range: [0.295, 0.405], tickvals: [0.30, 0.35, 0.40], constrain: 'domain' },
      yaxis: { title: { text: 'Predicted Benchmark Score', font: { size: 12 } }, gridcolor: '#eee', zeroline: false, range: [0.295, 0.505], tickvals: [0.30, 0.35, 0.40, 0.45, 0.50] },
      annotations: [{ xref: 'paper', yref: 'paper', x: 0.97, y: 0.06, text: 'R<sup>2</sup> = 0.664<br>RMSE = 0.440', showarrow: false, align: 'right', bordercolor: INK, borderwidth: 1, borderpad: 4, bgcolor: '#fff', font: { size: 11 } }],
    }, CONFIG);
  }

  /* ---- Heuristic: model family. Spearman rank-correlation of teacher
     rankings across student base models (values from the paper figure). ---- */

  const famMount = document.getElementById('chart-family-corr');
  if (famMount) {
    const bases = ['OLMo 3 7B', 'Gemma 3 4B', 'Qwen 3 8B', 'Llama 3 8B'];
    const N = null;
    // rows = y (OLMo bottom -> Llama top), cols = x; upper triangle blank
    const z = [
      [1.00, N, N, N],
      [0.87, 1.00, N, N],
      [0.60, 0.65, 1.00, N],
      [0.63, 0.68, 0.57, 1.00],
    ];
    const labels = [
      ['1.00', '', '', ''],
      ['0.87**', '1.00', '', ''],
      ['0.60', '0.65', '1.00', ''],
      ['0.63', '0.68*', '0.57', '1.00'],
    ];
    const annotations = [];
    z.forEach((row, r) => row.forEach((v, c) => {
      if (v === null) return;
      annotations.push({ x: bases[c], y: bases[r], text: labels[r][c], showarrow: false,
        font: { size: 12, color: v > 0.8 ? '#fff' : INK } });
    }));

    Plotly.newPlot(famMount, [{
      type: 'heatmap', x: bases, y: bases, z,
      colorscale: [[0, '#eef1ff'], [1, '#254eff']], zmin: 0.5, zmax: 1.0,
      xgap: 3, ygap: 3, showscale: false, hoverongaps: false,
      hovertemplate: '%{y} vs %{x}<br>ρ = %{z:.2f}<extra></extra>',
    }], {
      margin: { l: 90, r: 10, t: 60, b: 8 },
      height: 300, font: FONT, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      xaxis: { side: 'top', tickangle: -45, tickfont: { size: 11 }, constrain: 'domain' },
      yaxis: { tickfont: { size: 11 }, scaleanchor: 'x', scaleratio: 1, constrain: 'domain' },
      annotations: annotations.concat([{ xref: 'paper', yref: 'paper', x: 0.98, y: 0.12, text: '** p&lt;0.01&nbsp;&nbsp;* p&lt;0.05', showarrow: false, align: 'right', font: { size: 10, color: MUTED } }]),
    }, CONFIG);
  }

  /* ---- Heuristic: amount of SFT examples. Student performance vs. number of
     SFT examples (values recovered from the data_scale_effect figure). ---- */

  const scaleMount = document.getElementById('chart-data-scale');
  if (scaleMount) {
    const x = [1000, 5000, 10000, 25000, 50000];
    // one color, distinguished by marker shape + line dash
    const SERIES = {
      German: { symbol: 'square', dash: 'solid', y: [0.496, 0.553, 0.639, 0.642, 0.643] },
      Arabic: { symbol: 'circle', dash: 'dash', y: [0.441, 0.541, 0.617, 0.615, 0.628] },
      Indonesian: { symbol: 'triangle-up', dash: 'dashdot', y: [0.451, 0.484, 0.557, 0.581, 0.597] },
    };
    const traces = Object.keys(SERIES).map((lang) => ({
      type: 'scatter', mode: 'lines+markers', name: lang,
      x, y: SERIES[lang].y,
      line: { color: BLUE, width: 2, dash: SERIES[lang].dash },
      marker: { symbol: SERIES[lang].symbol, size: 9, color: BLUE, line: { color: INK, width: 1 } },
      hovertemplate: lang + '<br>%{x} samples<br>perf: %{y:.3f}<extra></extra>',
    }));

    Plotly.newPlot(scaleMount, traces, {
      margin: { l: 46, r: 12, t: 8, b: 40 },
      height: 268, font: FONT, paper_bgcolor: '#fff', plot_bgcolor: '#fff',
      legend: { orientation: 'h', y: -0.22, font: { size: 11 } },
      xaxis: {
        type: 'log', gridcolor: '#eee',
        tickvals: [1000, 10000], ticktext: ['1k', '10k'],
        title: { text: 'Number of SFT examples (log)', font: { size: 12 } },
      },
      yaxis: { gridcolor: '#eee', title: { text: 'Avg. multilingual performance', font: { size: 12 } } },
      shapes: [{ type: 'line', x0: 10000, x1: 10000, y0: 0, y1: 1, yref: 'paper', line: { color: MUTED, width: 1.5, dash: 'dot' } }],
      annotations: [{ x: Math.log10(10000), y: 0.46, text: 'gains flatten', showarrow: false, xanchor: 'left', font: { size: 10, color: MUTED } }],
    }, CONFIG);
  }

  /* ---- Tagalog case study: step-through ablation (values from the paper's
     tgl_ablation_filbench_scores figure). Each Next reveals one more bar. ---- */

  const ablMount = document.getElementById('chart-ablation');
  if (ablMount) {
    const GREY = '#c9c2b6', YMIN = 45;
    const bars = {
      x: ['Public', 'GPT-4o', 'Aya<br>Exp.', 'Match<br>family', '+25k<br>data', '12B<br>model', '27B<br>model'],
      y: [47.2, 47.7, 48.2, 49.5, 49.7, 51.4, 53.0],
      color: [GREY, GREY, BLUE, BLUE, BLUE, BLUE, BLUE],
    };
    const STEP = [
      { title: 'Baseline: publicly available data', prose: 'We start with a Gemma 3 4B student finetuned on 10k Tagalog prompt-response pairs sampled from public datasets, a non-synthetic baseline (10K-Public).' },
      { title: 'Use a synthetic pipeline', prose: 'Swapping public data for 10k instances synthesized by an off-the-shelf GPT-4o-mini teacher barely changes performance (about 0.5pp), suggesting there is no significant advantage to a synthetic pipeline if the teacher model is not optimal.' },
      { title: 'Use a better teacher', prose: 'We swap GPT-4o-mini for Aya Expanse 32B, a teacher with a higher PG-Score (0.706 vs. 0.461). The slight improvement suggests that PG-Score is generalizable to an unseen language.' },
      { title: 'Match teacher &amp; student families', prose: 'We use a Gemma 3 27B teacher to match the Gemma 3 4B student family. This yields a substantial improvement, demonstrating that family alignment is a reliable heuristic for teacher selection.' },
      { title: 'Scale the data (10k &rarr; 25k)', prose: 'Increasing synthetic instances from 10k to 25k gives a modest gain (+0.21pp), smaller than teacher selection or family matching. This is consistent with diminishing returns past 10k, though FilBench&rsquo;s diverse tasks suggest saturation is task-dependent.' },
      { title: 'Scale the student (4B &rarr; 12B)', prose: 'Scaling the student model from 4B to 12B parameters raises performance further, showing that the recipe benefits from increased model capacity.' },
      { title: 'Scale the student (12B &rarr; 27B)', prose: 'Scaling further to 27B continues the gains, yet our 25k-instance, SFT-only recipe stays data- and resource-efficient compared to heavier post-training pipelines.' },
    ];

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let step = 0;
    const yFor = (s) => bars.y.map((v, i) => (i <= s ? v : YMIN));
    const textFor = (s) => bars.y.map((v, i) => (i <= s ? v.toFixed(1) : ''));
    const trace = (s) => ({
      type: 'bar', x: bars.x, y: yFor(s), text: textFor(s),
      textposition: 'outside', textfont: { size: 12, color: INK }, cliponaxis: false,
      marker: { color: bars.color, line: { color: INK, width: 1 } }, hoverinfo: 'skip',
    });
    const layout = {
      margin: { l: 40, r: 10, t: 18, b: 40 }, height: 320, font: FONT,
      paper_bgcolor: '#fff', plot_bgcolor: '#fff', bargap: 0.34,
      xaxis: { tickfont: { size: 10.5 } },
      yaxis: { range: [YMIN, 54], gridcolor: '#eee', title: { text: 'FilBench Score', font: { size: 12 } } },
    };
    Plotly.newPlot(ablMount, [trace(0)], layout, CONFIG);

    const titleEl = document.getElementById('abl-title');
    const proseEl = document.getElementById('abl-prose');
    const stepEl = document.getElementById('abl-step');
    const backBtn = document.getElementById('abl-back');
    const nextBtn = document.getElementById('abl-next');

    function render() {
      const data = { data: [trace(step)] };
      if (reduceMotion) Plotly.react(ablMount, data.data, layout, CONFIG);
      else Plotly.animate(ablMount, data, { transition: { duration: 450, easing: 'cubic-out' }, frame: { duration: 450, redraw: false } });
      titleEl.innerHTML = STEP[step].title;
      proseEl.innerHTML = STEP[step].prose;
      stepEl.textContent = step === 0 ? 'Baseline' : 'Intervention ' + step + ' of 6';
      backBtn.disabled = step === 0;
      nextBtn.disabled = step === 6;
    }
    render();
    nextBtn.addEventListener('click', () => { if (step < 6) { step++; render(); } });
    backBtn.addEventListener('click', () => { if (step > 0) { step--; render(); } });
  }
})();
