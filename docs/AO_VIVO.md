# Jogos ao vivo (futebol) — Cloudflare Worker

A página mostra o minuto e o resultado dos jogos de futebol a decorrer. A chave da API-Football não pode ir para a
página (qualquer visitante a copiava), por isso um Cloudflare Worker grátis (`worker/live.js`) guarda a chave e
responde à página.

- 1 pedido (`fixtures?live=all`) traz todos os jogos ao vivo do mundo.
- O Worker guarda a resposta 20 minutos (KV): no máximo 72 pedidos por dia, só quando há visitantes, e deixa de
  pedir quando restam menos de 5. O job de hora a hora fica com 40 pedidos por dia (`RESERVE = 60` em
  `backend/app/data/api_football.py`).
- A página pergunta ao Worker a cada minuto; os resultados têm até 20 minutos de atraso.

## Configuração (uma vez)

1. Conta grátis em https://dash.cloudflare.com/sign-up
2. **Storage & Databases → KV → Create**: nome `LIVE`.
3. **Workers & Pages → Create → Worker**: nome `analise-live` → Deploy → **Edit code**: colar `worker/live.js` →
   Deploy.
4. No Worker → **Settings → Bindings → Add → KV namespace**: variável `LIVE`, namespace `LIVE`.
5. No Worker → **Settings → Variables and Secrets → Add → Secret**: nome `API_FOOTBALL_KEY`, valor = a chave da
   API-Football (só o dono da conta a cola).
6. O endereço do Worker (`https://analise-live.shadowteste2022.workers.dev`) está em `LIVE_URL` em `frontend/index.html`.

Atualizar o código do Worker: `cd worker && npx wrangler deploy` (configuração em `worker/wrangler.toml`).
