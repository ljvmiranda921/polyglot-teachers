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
})();
