// Monte Carlo bankroll simulation, run in the browser so the static site needs no server.
// Always shows the "model is wrong" scenarios next to the model's own estimate.
(function (root) {
  const LIMITS = {bankroll: [0, 1e9], stake_pct: [0, 0.5], n_bets: [1, 5000], prob: [0, 1], odds: [1, 1000], n_sims: [100, 20000]};

  function validate(p) {
    for (const [k, [lo, hi]] of Object.entries(LIMITS)) {
      const v = p[k];
      const ok = Number.isFinite(v) && (k === 'n_bets' || k === 'n_sims' ? v >= lo && v <= hi && Number.isInteger(v)
        : k === 'prob' ? v > lo && v < hi : v > lo && v <= hi);
      if (!ok) throw new Error(`Valor inválido para ${k}: ${v}`);
    }
  }

  // linear interpolation, same as numpy.percentile's default
  function percentile(sorted, q) {
    const i = (sorted.length - 1) * q / 100, lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }

  // mulberry32: seeded so results are reproducible for the same inputs
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const r = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

  function simulate({bankroll = 100, stake_pct = 0.01, n_bets = 500, prob, odds, n_sims = 3000, seed = 1}) {
    const params = {bankroll, stake_pct, n_bets, prob, odds, n_sims};
    validate(params);
    const random = rng(seed);
    const scenarios = {'estimativa do modelo': prob, 'sem vantagem (EV 0)': 1 / odds, 'modelo sobrestima (EV -5%)': 0.95 / odds};
    const step = Math.max(1, Math.floor(n_bets / 100));
    const out = {};
    for (const [name, p] of Object.entries(scenarios)) {
      const finals = [], dds = [], streaks = [], paths = [];
      let lose = 0, half = 0, ruin = 0;
      for (let s = 0; s < n_sims; s++) {
        let bank = bankroll, peak = bankroll, dd = 0, low = bankroll, cur = 0, longest = 0;
        const path = s < 20 ? [] : null;
        for (let b = 0; b < n_bets; b++) {
          const win = random() < p;
          bank *= win ? 1 + stake_pct * (odds - 1) : 1 - stake_pct;
          cur = win ? 0 : cur + 1;
          if (cur > longest) longest = cur;
          if (bank > peak) peak = bank;
          if ((peak - bank) / peak > dd) dd = (peak - bank) / peak;
          if (bank < low) low = bank;
          if (path && b % step === 0) path.push(r(bank, 2));
        }
        finals.push(bank); dds.push(dd); streaks.push(longest);
        if (bank < bankroll) lose++;
        if (low <= bankroll / 2) half++;
        if (low <= bankroll * 0.1) ruin++;
        if (path) paths.push(path);
      }
      [finals, dds, streaks].forEach(a => a.sort((x, y) => x - y));
      out[name] = {
        win_prob: r(p, 4), ev_per_unit: r(p * odds - 1, 4),
        final_p5: r(percentile(finals, 5), 2), final_median: r(percentile(finals, 50), 2),
        final_p95: r(percentile(finals, 95), 2),
        prob_loss: r(lose / n_sims, 4), prob_lose_half: r(half / n_sims, 4), prob_ruin_90: r(ruin / n_sims, 4),
        max_drawdown_median: r(percentile(dds, 50), 4),
        longest_losing_streak_median: Math.trunc(percentile(streaks, 50)),
        longest_losing_streak_p95: Math.trunc(percentile(streaks, 95)),
        sample_paths: paths};
    }
    const warnings = [];
    if (stake_pct > 0.05) warnings.push('Stakes acima de 5% da banca aumentam muito o risco de ruína, mesmo com vantagem real.');
    if (out['sem vantagem (EV 0)'].prob_loss > 0.4) warnings.push('Se o modelo não tiver vantagem real, é provável terminar com perdas.');
    warnings.push('Use apenas dinheiro que pode perder. Nunca aumente stakes para recuperar perdas.');
    return {params, scenarios: out, warnings};
  }

  root.simulateRisk = simulate;
  if (typeof module !== 'undefined') module.exports = {simulate, percentile};
})(typeof window !== 'undefined' ? window : globalThis);
