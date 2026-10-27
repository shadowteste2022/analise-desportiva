// Cloudflare Worker: live football scores for the site, without exposing the API-Football key.
// Setup (Cloudflare dashboard): secret API_FOOTBALL_KEY, KV namespace bound as LIVE. See docs/AO_VIVO.md.
//
// One request to API-Football (fixtures?live=all) returns every live game in the world. The answer is kept in KV for
// TTL_S seconds, so however many visitors there are, the free quota (100 requests/day, shared with the hourly job)
// is spent at most once per TTL_S; below MIN_REMAINING requests left, the last saved answer is served instead.
const TTL_S = 20 * 60;
const MIN_REMAINING = 5;
const ALLOWED = ["https://shadowteste2022.github.io", "http://localhost:8090", "http://localhost:8000"];

function cors(origin) {
  return {
    "Access-Control-Allow-Origin": ALLOWED.includes(origin) ? origin : ALLOWED[0],
    "Access-Control-Allow-Methods": "GET",
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "public, max-age=60",
  };
}

function compact(response) {
  return response.map((g) => ({
    home: g.teams.home.name, away: g.teams.away.name,
    league: g.league.name, country: g.league.country,
    status: g.fixture.status.short, elapsed: g.fixture.status.elapsed, extra: g.fixture.status.extra,
    hg: g.goals.home, ag: g.goals.away, kickoff: g.fixture.date,
  }));
}

export default {
  async fetch(request, env) {
    const headers = cors(request.headers.get("Origin") || "");
    if (request.method === "OPTIONS") return new Response(null, { headers });
    const saved = await env.LIVE.get("live", "json");  // {at, games, remaining}
    const now = Date.now();
    if (saved && now - saved.at < TTL_S * 1000) return new Response(JSON.stringify(saved), { headers });
    if (saved && saved.remaining != null && saved.remaining < MIN_REMAINING && new Date(saved.at).getUTCDate() === new Date(now).getUTCDate()) {
      return new Response(JSON.stringify({ ...saved, stale: true }), { headers });  // quota nearly used up today
    }
    try {
      const r = await fetch("https://v3.football.api-sports.io/fixtures?live=all", {
        headers: { "x-apisports-key": env.API_FOOTBALL_KEY },
      });
      const body = await r.json();
      if (!r.ok || (body.errors && Object.keys(body.errors).length)) throw new Error(JSON.stringify(body.errors || r.status));
      const remaining = Number(r.headers.get("x-ratelimit-requests-remaining"));
      const fresh = { at: now, games: compact(body.response || []), remaining: Number.isFinite(remaining) ? remaining : null };
      await env.LIVE.put("live", JSON.stringify(fresh));
      return new Response(JSON.stringify(fresh), { headers });
    } catch (e) {
      // the source failed: the last saved answer (marked stale) beats nothing
      return new Response(JSON.stringify(saved ? { ...saved, stale: true } : { at: now, games: [], error: String(e) }), { headers });
    }
  },
};
