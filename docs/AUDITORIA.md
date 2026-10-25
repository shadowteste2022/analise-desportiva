# Auditoria do projeto — 25/10/2026

Estado auditado: commit `071334d`, site publicado (gerado 25/10 11:20 UTC), últimas 23 execuções do workflow.
Tudo o que está abaixo foi medido nesta auditoria; nada é estimado sem o dizer.

## 1. O que funciona

| Área | Evidência |
|---|---|
| Testes | 58 testes Python e 4 JS passam; `pip check` sem conflitos |
| Pipeline | 21/23 execuções com sucesso (as 2 falhas foram corrigidas em commits seguintes) |
| Modelo | Dixon-Coles com decaimento temporal, combinação modelo+mercado ajustada fora da amostra, Shin para retirar a margem |
| Backtest | Walk-forward em 8 épocas e 21 ligas; CLV medido contra o fecho da Pinnacle |
| Honestidade | Barreira de validação: nenhuma liga bate o fecho, por isso nunca aparece «VALOR POTENCIAL» |
| Filtros de odds | Outliers (15%), divergência face ao mercado (12%), exchanges excluídas (comissão), jogos já começados escondidos |
| Odds legais PT | OddsPapi (Betano PT) funciona localmente: 173 jogos, 7 mercados, 48 de 103 jogos football-data emparelhados |
| Quotas | Odds API: 88/500 créditos usados com ritmo por período; OddsPapi: 21/250 pedidos |
| Segredos | Nenhuma chave no repositório nem no site; `.env` ignorado pelo git |

## 2. O que está incompleto

- **OddsPapi não corre no site.** O site publicado mostra `legal.sources: []`: o secret `ODDSPAPI_KEY` está vazio ou em falta no GitHub (o workflow já o passa). Por isso, **0 das 99 melhores apostas estão numa casa licenciada SRIJ**; dominam Bet365 (63) e Betway (21), que não são legais em Portugal.
- **PT_ONLY desligado** — depende do ponto anterior.
- **125 jogos só da Betano** (MLS, Brasil, Champions feminina…) entram sem probabilidades: com uma só casa não há consenso de mercado.
- **203 de 302 eventos (67%) estão «SEM DADOS SUFICIENTES»**, sobretudo outras modalidades com poucas casas.
- **Mercados de futebol:** só 1X2, golos (0.5–4.5) e ambas marcam. Faltam dupla hipótese e empate anula (deriváveis já do modelo), handicaps, golos por equipa, cantos e cartões (há dados nos CSV: HC/AC, HY/AY, HR/AR), remates. Props de jogadores: não há fonte gratuita.
- **Desempenho real** (`history.json`) existe mas não mostra ainda métricas agregadas (log loss, Brier, ROI, CLV ao vivo) por mercado/liga.

## 3. O que é simulado

- **Backtest:** é real (resultados e odds históricas), mas é uma *simulação de apostas* passadas — não são apostas feitas.
- **Simulador de risco** (página Risco): simulação Monte Carlo explícita, assim identificada.
- **Não há odds, resultados nem métricas inventadas.** Odds de exemplo só existem nos testes.

## 4. Erros e riscos

| Gravidade | Problema |
|---|---|
| Alta | Melhor aposta recomendada em casas **não licenciadas em PT** (ver ponto 2) |
| Alta | **«Hora a hora» não é real:** o cron do GitHub é de melhor esforço — as execuções agendadas de hoje foram às 23:09, 01:06, 06:03 e 11:19 (intervalos até 5 h). Odds podem ficar desatualizadas horas |
| Média | A chave OddsPapi foi colada no chat: recomenda-se gerá-la de novo no painel da OddsPapi |
| Média | Base de dados de previsões vive na cache do Actions + commit diário; se a cache for apagada perde-se até 1 dia de snapshots |
| Média | A barreira de validação é a do futebol; nas outras modalidades a «melhor opção» é só a melhor odd face ao consenso, sem backtest (o site já o diz, mas o rótulo pode ser lido como conselho) |
| Baixa | 1 jogo começado há mais de 2 h ainda no ficheiro (consequência do atraso do cron) |
| Baixa | `starlette` 1.9.0 → 1.10.0 disponível |

## 5. O que falta (pedido no prompt mestre)

Market Scanner, comparação de mercados lado a lado, histórico de movimento de odds em gráfico, pontuação de **qualidade da evidência** separada da probabilidade, «odd mínima para ter valor», motor de melhor aposta ponderado por incerteza, alertas configuráveis, página de desempenho com métricas.

## 6. Melhorias prioritárias

1. Pôr o secret `ODDSPAPI_KEY` no GitHub → odds Betano PT no site; depois `PT_ONLY=true`.
2. Novos mercados derivados do modelo sem custo de dados: dupla hipótese, empate anula, golos por equipa, handicap asiático simples.
3. Qualidade da evidência (0–100) por análise: nº de casas, jogos de histórico das equipas, idade das odds, validação da liga, divergência modelo/mercado.
4. Motor de melhor aposta: ordenar por limite inferior do EV (EV − incerteza), não pelo EV bruto; quando nada passa, mostrar «Não foi identificada uma oportunidade com valor estatístico suficientemente convincente nas condições atuais.»
5. Odd mínima justa por seleção (1 / probabilidade × (1 + margem exigida)).
6. Desempenho ao vivo: log loss, Brier, CLV e ROI simulado das previsões já registadas.
7. Cantos/cartões com modelo Poisson próprio (dados nos CSV) — só se o backtest mostrar calibração aceitável.

## 7. Diferenciação face à concorrência

| Plataforma | Foco | O que não faz e nós podemos fazer |
|---|---|---|
| FBref / Understat | Estatística avançada (xG Opta; Understat: 6 ligas) | Não liga estatística a odds nem a valor |
| Sofascore / FotMob / Flashscore | Resultados ao vivo, ratings, xG | Odds só de parceiros; nenhuma avaliação de valor |
| WhoScored / Transfermarkt / Soccerway | Ratings, valores de mercado, arquivo | Sem probabilidades nem odds |
| OddsPortal | Comparação de 60+ casas, *dropping odds*, *value bets* contra a média | Casas internacionais, sem filtro SRIJ; «valor» = desvio da média, sem modelo nem validação |
| Comparadores PT (ex.: MelAp) | Bónus e casas SRIJ | Afiliação; sem análise estatística |

**Espaço livre:** nenhum dos analisados junta (a) só casas licenciadas SRIJ, (b) probabilidades com margem retirada e modelo validado fora da amostra, (c) honestidade explícita — dizer «sem valor» quando não há, com qualidade da evidência e desempenho real publicado.

## 8. Plano ordenado

1. **Odds legais** — secret `ODDSPAPI_KEY` (utilizador), confirmar no site, ligar `PT_ONLY`.
2. **Mercados derivados** do modelo (dupla hipótese, empate anula, golos por equipa, handicap).
3. **Qualidade da evidência + motor de melhor aposta** com incerteza e mensagem de «sem oportunidade».
4. **Odd mínima** e comparação de mercados na página do jogo.
5. **Desempenho ao vivo** (métricas das previsões registadas) e gráfico de movimento de odds.
6. **Atualização mais fiável:** avaliar disparo externo do workflow (ex.: cron-job.org a chamar `workflow_dispatch`) para cumprir a hora.
7. Cantos/cartões, depois de backtest.

## Fontes da pesquisa de mercado

- FBref — estatística avançada: https://info.sports-reference.com/migration/knowledge/soccer-advanced-stats-explained
- Understat (perfil): https://www.stork.ai/en/understat
- Lista de ferramentas de estatística: https://zipdo.co/best/football-stat-software/
- OddsPortal *dropping odds*: https://www.oddsportal.com/dropping-odds/2/2/0/football/
- Comparadores de odds: https://rapidapi.com/blog/sports-odds-sites/
- Comparador PT (MelAp): https://www.gazetaesportiva.com/?p=2871538
