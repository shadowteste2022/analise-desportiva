// node --test frontend/tests/risk.test.js
const test = require('node:test');
const assert = require('node:assert');
const {simulate, percentile} = require('../static/risk.js');

test('scenarios are ordered and probabilities consistent', () => {
  const r = simulate({bankroll: 100, stake_pct: 0.02, n_bets: 200, prob: 0.55, odds: 2.0, n_sims: 2000, seed: 3});
  const s = r.scenarios;
  assert.strictEqual(s['sem vantagem (EV 0)'].ev_per_unit, 0);
  assert.ok(s['estimativa do modelo'].final_median > s['modelo sobrestima (EV -5%)'].final_median);
  for (const v of Object.values(s)) assert.ok(v.prob_ruin_90 >= 0 && v.prob_ruin_90 <= v.prob_lose_half && v.prob_lose_half <= 1);
  assert.strictEqual(s['estimativa do modelo'].sample_paths.length, 20);
});

test('same seed gives the same result', () => {
  const p = {prob: 0.4, odds: 2.6, n_bets: 50, n_sims: 300};
  assert.deepStrictEqual(simulate(p), simulate(p));
});

test('rejects invalid input', () => {
  assert.throws(() => simulate({prob: 1.5, odds: 2}), /prob/);
  assert.throws(() => simulate({prob: 0.5, odds: 0.9}), /odds/);
  assert.throws(() => simulate({prob: 0.5, odds: 2, n_bets: 2.5}), /n_bets/);
});

test('percentile matches numpy linear interpolation', () => {
  assert.strictEqual(percentile([1, 2, 3, 4], 50), 2.5);
  assert.strictEqual(percentile([0, 10], 5), 0.5);
});
