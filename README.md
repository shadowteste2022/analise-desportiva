# Análise Desportiva — MVP Futebol

Plataforma de análise estatística pré-jogo para futebol: recolhe dados reais, estima probabilidades com um modelo
Dixon-Coles, compara com as odds de vários operadores, calcula o valor esperado (EV) e classifica cada mercado.
Inclui backtesting walk-forward e um registo de todas as previsões. A interface é uma página única só de análise e previsão (sem odds nem casas de apostas): destaques do dia, a lista de
jogos com o resultado mais provável e, dentro de cada jogo, todas as probabilidades — resultado, golos, intervalo,
intervalo/final, margem, remates, cantos, cartões e resultado exato. As odds continuam a ser usadas por dentro, como
referência de mercado que torna as probabilidades mais precisas.

> **Não garante resultados nem lucros.** É uma ferramenta de análise e simulação; não faz apostas.
> O próprio backtest (8 épocas, abaixo) mostra que nem o modelo nem a combinação modelo + mercado **superam as odds
> de fecho** em nenhuma das 7 ligas.
> Uso exclusivo para maiores de 18 anos. Ajuda em Portugal: SICAD 1414.

## Executar localmente (Windows)

Requisitos: Python 3.12.

```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r backend/requirements.txt
```
```bash
cp .env.example .env
```
```bash
.venv/Scripts/python -m uvicorn app.main:app --port 8000 --app-dir backend
```

Abra http://localhost:8000. O primeiro carregamento descarrega cerca de 21 CSV (≈20 s); depois tudo fica em cache.

Correr o backtest (≈10 min; gera `data/backtests/<LIGA>.json`, que a interface mostra e que **valida** cada liga,
e `data/blend.json`, com os pesos da combinação modelo + mercado):

```bash
cd backend && ../.venv/Scripts/python -m app.backtest
```

Testes (40 em Python e 4 em JavaScript, com dados sintéticos e sem rede):

```bash
cd backend && ../.venv/Scripts/python -m pytest -q
```

Em Linux/macOS use `.venv/bin/python` em vez de `.venv/Scripts/python`.

## Publicar grátis no GitHub Pages (atualização de hora a hora)

O site publicado é estático: de hora a hora, um GitHub Action corre a mesma análise da API, grava o resultado em
ficheiros JSON e publica-os com a página. Não há servidor nem chaves expostas. A página verifica dados novos a
cada 5 minutos.

```
.github/workflows/daily.yml   de hora a hora (e a pedido): análises → Pages
                               uma vez por dia (07 UTC) ou a pedido: também testes e commit de state/
backend/app/export.py          gera site/ (index.html + data/*.json + static/)
state/                         o que tem de sobreviver entre execuções: app.db (histórico de previsões),
                               blend.json (pesos da combinação) e backtests/*.json
```

**Como o estado passa de hora para hora:** a base de dados, as odds guardadas e os CSV em cache passam de uma
execução para a seguinte pela cache do GitHub Actions. Uma vez por dia, `state/` é gravado no repositório, para o
histórico sobreviver mesmo que a cache seja apagada. Gravar de hora a hora um ficheiro SQLite faria o repositório
crescer vários GB por ano. Para manter a base pequena, os snapshots de odds só são gravados quando a odd muda, e os
de jogos com mais de 14 dias são apagados. O histórico de previsões fica completo.

### Odds ao vivo sem passar do plano (The Odds API)

Com o secret `ODDS_API_KEY`, cada execução vai buscar odds ao vivo, mas **com controlo de quota**:
- **Orçamento repartido:** o orçamento mensal (`ODDS_API_MONTHLY_BUDGET`, por defeito 500, igual ao plano grátis) é
  distribuído ao longo do período de faturação. Uma chamada só é feita se o gasto, incluindo essa chamada, couber na
  parte do orçamento já "ganha" até esse momento, mais a folga de uma atualização completa (2 créditos × número de
  ligas). Nunca desce abaixo da reserva `ODDS_API_MIN_REMAINING` (20).
- **Consumo real:** é lido no endpoint `/sports` da The Odds API, que não gasta créditos. Se não for possível ler o
  consumo, não é feita nenhuma chamada.
- **Prioridade:** as ligas são pedidas pela ordem dos jogos, por isso as que jogam mais cedo são atualizadas primeiro.
- **Ligas adiadas:** quando o orçamento não chega, a liga mostra as últimas odds guardadas (até
  `ODDS_MAX_AGE_HOURS` = 48 h), e a página indica os créditos usados e as ligas adiadas.
- **Ciclo de faturação:** se o plano renova noutro dia do mês, defina `ODDS_API_RESET_DAY`.

Simulação de um mês com 7 ligas e o plano grátis: foram gastos 480 dos 500 créditos, cerca de 16 por dia, o que
corresponde a **cada liga ser atualizada mais ou menos uma vez por dia**. Correr de hora a hora não aumenta o gasto,
só o distribui. Para atualizar cada liga de hora a hora são precisos cerca de 10 000 créditos por mês (plano pago):
basta mudar a variável `ODDS_API_MONTHLY_BUDGET`, sem mexer no código.

Passos (uma vez):
1. Crie um repositório no GitHub. Tem de ser **público**: no plano gratuito, o GitHub Pages só funciona com
   repositórios públicos. Isso torna visíveis o código e o histórico de previsões (`state/app.db`), que não contém
   dados pessoais.
2. Envie o projeto:
   ```bash
   git init
   ```
   ```bash
   git add . && git commit -m "Primeira versão"
   ```
   ```bash
   git branch -M main && git remote add origin https://github.com/<utilizador>/<repo>.git && git push -u origin main
   ```
   O `.gitignore` deixa de fora o `.env`, `.venv/`, `data/` (dados locais) e `site/`.
3. No GitHub, em **Settings → Pages → Build and deployment → Source**, escolha **GitHub Actions**.
4. Em **Settings → Secrets and variables → Actions**:
   - **Secrets:** `ODDS_API_KEY` (crie uma conta grátis em the-odds-api.com) e, opcionalmente, `ODDS_API_IO_KEY` para
     as odds das casas SRIJ.
   - **Variables (opcional):** `ODDS_API_MONTHLY_BUDGET` se tiver um plano pago, e `ODDS_API_RESET_DAY` se o plano
     não renovar no dia 1.
5. Em **Actions → Análises de hora a hora → Run workflow**, faça a primeira publicação. O site fica em
   `https://<utilizador>.github.io/<repo>/`.

Notas:
- **Custo do GitHub:** em repositórios públicos, os minutos do Actions e o Pages são grátis. Cada execução demora
  cerca de 2 minutos.
- **Agendamento:** o GitHub pode atrasar ou saltar execuções agendadas em alturas de muita carga, por isso "de hora
  a hora" é aproximado. Também desativa execuções
  agendadas em repositórios públicos sem atividade durante 60 dias, e os commits automáticos podem não contar como
  atividade. Se isso acontecer, o GitHub avisa por email; basta reativar em **Actions**.
- **Backtest:** não corre todos os dias, porque demora cerca de 10 minutos. Para o atualizar, corra-o localmente
  com `DATA_DIR=state` e faça commit de `state/`. Uma vez por mês chega.
- **Jogos fora da lista:** um jogo que já saiu da lista de próximos jogos deixa de ter página própria, mas as suas
  previsões ficam no Histórico.

Para testar o site estático localmente antes de publicar:
```bash
cd backend && ../.venv/Scripts/python -m app.export --out ../site
```
```bash
.venv/Scripts/python -m http.server 8001 --directory site
```

## O que precisa de serviços externos

| Componente | Fonte | Chave? | Custo | Estado |
|---|---|---|---|---|
| Resultados, golos, xG, remates, cantos, cartões, odds históricas e de fecho | [football-data.co.uk](https://www.football-data.co.uk) (CSV) | Não | Grátis | **Integrado e testado com dados reais** |
| Próximos jogos com odds pré-jogo de 6 operadores (1X2, O/U 2,5), 21 ligas | football-data.co.uk `fixtures.csv` | Não | Grátis | **Integrado e testado com dados reais** |
| Odds ao vivo, mais operadores, outras linhas de golos | [The Odds API](https://the-odds-api.com) | `ODDS_API_KEY` | Grátis ~500 créditos/mês; planos pagos acima | Integrado; testado no GitHub até ao pedido (resposta: chave inválida) |
| **Outras modalidades**: NBA, Euroleague, NHL, MLB, torneios ATP/WTA ativos | The Odds API | `ODDS_API_KEY` | Partilha o orçamento acima | Integrado; só aparecem com uma chave válida |
| **Odds de casas licenciadas em Portugal** (Betano PT, Betclic PT, Bwin PT): 1X2, totais 0,5–4,5, BTTS | [odds-api.io](https://odds-api.io) | `ODDS_API_IO_KEY` | Desde £49/mês (plano Solo = 2 casas); novas chaves grátis suspensas | Integrado; **só testado com resposta simulada** (não tenho chave) |
| Lesões, suspensões, onzes prováveis | ex.: API-Football (api-sports.io), Sportmonks | Sim | Pago acima do plano grátis | **Não integrado**: cada relatório avisa disto |
| Meteorologia | ex.: Open-Meteo | Não | Grátis | Não integrado |

Os preços são os conhecidos à data do desenvolvimento; confirme-os antes de contratar. As chaves ficam só no backend
(`.env`) e nunca são enviadas ao browser nem escritas nos logs.

## Ligas e modalidades

**Futebol (21 ligas do football-data):** Portugal; Inglaterra (Premier League, Championship, League One, League
Two, National League); Espanha, Itália, Alemanha e França (1.ª e 2.ª divisões); Países Baixos; Bélgica; Turquia;
Grécia; Escócia (Premiership, Championship, League One). Configurável em `LEAGUES`.

O `fixtures.csv` só é republicado cerca de duas vezes por semana e mantém os jogos já disputados até lá. A app esconde
os jogos que começaram há mais de 2 horas (hora de Lisboa, que é a mesma do Reino Unido usada pela fonte).

**Futebol, mais 16 ligas** (formato "extra" do football-data, com o mesmo modelo e estatísticas): Brasil,
Argentina, EUA (MLS), México, Japão, China, Áustria, Suíça, Dinamarca, Noruega, Suécia, Finlândia, Polónia, Roménia,
Irlanda e Rússia. O histórico destas ligas só tem odds de fecho, por isso ficam fora do backtest e nunca são validadas.

**Todas as outras modalidades e competições (`OTHER_SPORTS=*`):** cada execução pede à The Odds API a lista de
competições em época (grátis). Entram todas, masculinas e femininas: futebol de todo o mundo (Champions League, ligas
femininas…), basquetebol, hóquei, basebol, ténis ATP/WTA, futebol americano, râguebi, críquete, MMA, boxe, etc. Só
ficam de fora os mercados de vencedor final (golfe, apostas de longo prazo) e a política.
- **Jogos:** vêm do endpoint `/events`, que não gasta créditos, por isso **aparecem todos**.
- **Odds:** gastam créditos. São pedidas competição a competição, a começar pelo jogo mais próximo, até ao limite do
  orçamento. As modalidades sem outra fonte de odds têm prioridade sobre as ligas do football-data, que já trazem odds
  de 6 casas nos CSVs. Jogos sem odds aparecem como "sem odds".
- **Confirmação de que é grátis:** se o `/events` alguma vez gastar créditos (cabeçalho `x-requests-last`), deixa de
  ser chamado.
- **Probabilidades:** não há fonte grátis com histórico de resultados e odds como o football-data, por isso ainda não
  há modelo estatístico próprio. As probabilidades são o **consenso das casas sem margem** (Shin sobre a odd média).
- **Valor:** o "valor" mostrado é a melhor odd face a esse consenso. Sem backtest, nunca aparece `VALOR POTENCIAL`.
- **Orçamento:** estas modalidades partilham o orçamento da The Odds API e são pedidas depois do futebol.

## Casas de apostas legais em Portugal

Só operadores licenciados pelo [SRIJ](https://www.srij.turismodeportugal.pt/pt/jogo-online/entidades-licenciadas/)
podem aceitar apostas de Portugal. Lista de licenças de apostas desportivas, consultada a 24/10/2026 (página atualizada
a 19/10/2026; confirme sempre no SRIJ, porque as licenças mudam):

| Marca | Licença | Odds na app |
|---|---|---|
| Betclic (betclic.pt) | 001 | ✔ via odds-api.io (Betclic PT) |
| Bwin (bwin.pt) | 002 | ✔ via odds-api.io (Bwin PT) |
| Betano (betano.pt) | 018 | ✔ via odds-api.io (Betano PT) |
| Placard.pt | 013 | ✘ sem feed legal conhecido |
| Casino Solverde (solverde.pt) | 024 | ✘ |
| ESC Online (estorilsolcasinos.pt) | 008 | ✘ |
| Casino Portugal | 007 | ✘ |
| LeBull | 027 | ✘ |
| GoldenPark | 029 | ✘ |
| Retabet (retabet.pt) | 033 | ✘ (só há a versão espanhola nos feeds) |
| Nossa Aposta, Luckia, Moosh | 012, 014, 019 | atividade suspensa |

As casas sem feed não são extraídas dos sites, porque isso violaria os termos de utilização. A The Odds API só tem as
versões FR/UK da Betclic e da Betano, cujas odds não são as do .pt.

**Regra da melhor odd**: escolhe primeiro entre casas SRIJ. Se nenhuma tiver odd para o mercado, usa a melhor odd
internacional dos CSVs (Bet365, Betfred, etc., **sem licença em Portugal**) só como referência: fica marcada
"sem licença PT" e o motivo é indicado. O backtest histórico usa estas odds internacionais, porque não existem odds
históricas das casas portuguesas nas fontes gratuitas.

## Arquitetura

```
backend/app/
  config.py              configuração (.env)
  data/football_data.py  recolha, cache em disco, validação de CSVs
  data/odds_api.py       cliente opcional The Odds API (quota, cache, correspondência difusa de nomes)
  data/odds_api_io.py    cliente opcional odds-api.io (casas licenciadas SRIJ)
  models/dixon_coles.py  modelo de golos
  markets.py             probabilidades por mercado, remoção de margem, EV, classificação
  analysis.py            relatório A–E por jogo, validação por liga, histórico
  backtest.py            backtest walk-forward e métricas
  db.py                  SQLite: registo de previsões e snapshots de odds
  export.py              exportação estática para o GitHub Pages
  main.py                API FastAPI e serviço da interface
frontend/index.html      interface de página única (HTML + JS, sem dependências); funciona com a API ou com os JSON estáticos
```

**Decisões e desvios à stack proposta**, para manter o MVP pequeno e fácil de correr:
- **Interface numa página HTML servida pelo FastAPI**, em vez de Next.js: sem Node nem build, e um só processo.
  A API é JSON limpo, por isso um frontend Next.js pode ser acrescentado depois sem mudar o backend.
- **SQLite em vez de PostgreSQL** e **cache em disco/memória em vez de Redis**: chega para um utilizador. Mudar
  quando houver vários workers ou utilizadores (ver o comentário `shortcut:` em `db.py`).
- **Sem scikit-learn por agora**: o Dixon-Coles dá a distribuição completa de resultados de forma exata, por isso
  1X2, totais e BTTS saem da mesma distribuição sem contar a mesma informação duas vezes. Também não é preciso Monte
  Carlo para os mercados, porque a matriz de resultados é exata. Modelos de ML ficam para quando houver features que o
  modelo de golos não capta (xG, lesões).
- **Autenticação**: não há contas de utilizador nem dados pessoais. As operações de administração exigem o cabeçalho
  `X-Admin-Token` (comparado em tempo constante). Se for exposto à internet, é preciso autenticação real e HTTPS.

## Metodologia

- **Modelo**: Dixon-Coles (1997). Golos ~ Poisson(ataque, defesa, vantagem de casa), com correção ρ para 0-0/1-0/0-1/1-1,
  decaimento temporal ξ = 0,0019/dia (meia-vida ≈ 1 ano) e penalização ridge que puxa equipas com poucos jogos para a
  média. Treina apenas com jogos **anteriores** à data prevista.
- **Fórmulas**: probabilidade implícita = 1/odd; odd justa = 1/p; EV = p × odd − 1. A probabilidade do mercado sem
  margem obtém-se pelo **método de Shin** aplicado à odd média de todos os resultados (`DEVIG_METHOD`). Shin corrige o
  viés favorito-azarão, e o backtest confirma que dá probabilidades de mercado mais precisas do que a normalização
  proporcional. O mesmo método é usado na combinação com o mercado e no CLV contra a Pinnacle.
- **Melhor odd**: máximo entre operadores nomeados. Exclui agregados (máximo/média) e a Betfair Exchange (antes de comissão).
- **Confiança** (separada da probabilidade): baseia-se no número de jogos de cada equipa na janela de treino
  (<10 = sem previsão, <20 baixa, <40 média, resto alta).
- **Classificação**:
  - `SEM DADOS SUFICIENTES`: sem odd, sem histórico ou odds em cache com mais de `ODDS_MAX_AGE_HOURS`.
  - `SEM VALOR IDENTIFICADO`: EV ≤ 0.
  - `VALOR INCONCLUSIVO`: EV positivo, mas abaixo de `EV_MARGIN` (3%), ou confiança baixa, ou modelo a mais de 12 pp
    do mercado (sinal de informação que o modelo não vê), ou odd mais de 15% acima da odd justa do mercado (provável
    erro de dados), ou **liga sem validação**.
  - `VALOR POTENCIAL`: tudo o resto, **e só se o backtest dessa liga mostrar a probabilidade final tão precisa como
    as odds de fecho e com CLV médio positivo**.
- **Probabilidade final**: no 1X2 e no O/U 2,5, quando há odd média pré-jogo, é a combinação modelo + mercado com os
  pesos de `data/blend.json`, gerado pelo backtest. Nos outros mercados (outras linhas e BTTS) é só o Dixon-Coles,
  porque não há odds históricas desses mercados para validar a combinação. O relatório de cada jogo mostra as duas.
- **Mercados derivados**: dupla hipótese e empate anula saem da probabilidade final 1X2; golos por equipa (0,5–2,5) e
  handicap ±1,5 saem só da matriz do Dixon-Coles. Nenhuma fonte dá odds destes mercados, por isso aparecem com a
  probabilidade, a odd justa e a odd mínima, para comparar na casa. No empate anula, a odd justa usa a probabilidade
  condicionada a não haver empate e o EV desconta a devolução do stake.
- **Golos esperados calibrados**: quando há mercado para o resultado e para o O/U 2,5, procuram-se os golos esperados
  cuja grelha de resultados (com a correção do Dixon-Coles) reproduz essas probabilidades finais. Resultado exato,
  outras linhas de golos sem mercado, golos por equipa, handicap e intervalo saem dessa grelha, por isso batem certo
  com os números principais. Os golos esperados do modelo sozinho continuam no relatório (`model_expected_goals`).
- **Intervalo**: Poisson com os golos esperados × fração dos golos da liga marcados antes do intervalo.
- **Cartões amarelos**: média recente (último ano) de cada equipa em casa/fora, puxada para a média da liga como se
  fossem mais 10 jogos, e binomial negativa com a sobredispersão da liga. **Cantos**: frequência real da liga no
  último ano (as médias por equipa não melhoraram a previsão fora da amostra). O árbitro não entra.
- **Validação destes mercados** (época 2025/26, 8 ligas, sem dados futuros; referência = frequências da liga):
  resultado ao intervalo log loss 1,033 vs 1,080; golos ao intervalo +0,5 Brier 0,2032 vs 0,2039 e +1,5 0,2227 vs
  0,2253; cartões +3,5 0,2372 vs 0,2400 e +4,5 0,2227 vs 0,2255. Cantos por equipa: 0,2308 vs 0,2300 (pior) —
  por isso os cantos usam a frequência da liga. Nenhum destes mercados tem odds nas fontes gratuitas: sem EV.
- **Remates e remates à baliza**: o mesmo modelo dos cartões. Validação em 2025/26 (8 ligas): remates Brier
  0,2086 vs 0,2133 da frequência da liga; remates à baliza 0,2198 vs 0,2225 (melhor nas duas épocas testadas).
- **Golos exatos, baliza a zeros, margem de vitória** saem da grelha de resultados; **intervalo/final, 2.ª parte e
  parte com mais golos** de Poisson independente em cada parte (1.ª = fração da liga, 2.ª = o resto).
- **Cobertura máxima com dados grátis**: qualquer jogo com pelo menos uma casa com preço tem previsão (as odds dessa
  casa sem a margem; confiança «muito baixa» com uma só casa). Equipas sem histórico na liga (recém-promovidas) usam o
  mercado sem margem em vez do modelo. Nas outras modalidades a The Odds API só pede o resultado (1 crédito) e as odds
  são reutilizadas durante `ODDS_TTL_HOURS` (12 h), para os 500 créditos grátis chegarem a mais modalidades.
  Jogos sem nenhum preço aparecem na lista como «sem previsão».
- **Odd mínima** = odd justa × (1 + `EV_MARGIN`): abaixo dela a aposta não compensa a margem de segurança.
- **Qualidade da evidência** (0–100, separada da probabilidade): 4 partes de 25 — n.º de casas com preço, histórico das
  equipas, validação fora da amostra da liga e acordo entre o modelo e o mercado sem margem. «forte» exige liga validada.
- **Motor da melhor aposta**: entre as seleções verificáveis (há referência de mercado e nenhum alerta de erro de
  odd ou divergência), ordena pelo **EV conservador** — o menor entre o EV do modelo e o EV contra o mercado sem
  margem — multiplicado pela evidência quando é positivo. Sem `VALOR POTENCIAL`, a página diz «Não foi identificada
  uma oportunidade com valor estatístico suficientemente convincente nas condições atuais.» e mostra a melhor opção
  como não validada.
- **Desempenho ao vivo** (página Histórico): log loss, Brier, calibração e lucro simulado das seleções com EV positivo,
  usando a última estimativa registada de cada seleção.
- **Movimento das odds**: cada atualização guarda só os preços que mudaram, com a hora a que foram observados; a página
  do jogo mostra a evolução da melhor odd do resultado.

## Backtest: 8 épocas e modelo combinado com o mercado (24/10/2026)

As odds médias pré-jogo e de fecho só existem nos CSVs desde 2019/20. Por isso o teste vai de 2019/20 a 2026/27
(8 épocas), com 2017/18 e 2018/19 apenas como aquecimento.

- **Dixon-Coles**: reajustado todas as semanas só com jogos anteriores, com a mesma janela de 3 anos da app.
- **Combinação modelo + mercado**: pool logarítmico `p ∝ p_modelo^a · p_mercado^b`, com 2 pesos por mercado
  (1X2 e O/U 2,5). O mercado entra pela odd média **pré-jogo** sem margem, ou seja, o que se sabe no momento da
  aposta. O termo de comparação é a odd de **fecho**.
- **Fora da amostra**: os pesos usados na época S são ajustados só com previsões das épocas anteriores a S, juntando
  as 7 ligas. 2019/20 serve apenas para treinar a combinação e fica fora de todas as métricas.
- **Apostas**: o mesmo código de decisão da app, com stake de 1u.
- **Margem**: retirada com o método de Shin em todo o lado (mercado pré-jogo, fecho e Pinnacle).

```bash
cd backend && ../.venv/Scripts/python -m app.backtest
```

| Liga | Jogos | Final (combinado) | Só Dixon-Coles | Mercado pré-jogo | Mercado fecho | O/U 2,5 final / fecho | Apostas | Yield | IC 95% yield | CLV Pinnacle | IC 95% CLV | Acima do fecho |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 2342 | 0.9650 | 0.9839 | 0.9650 | 0.9607 | 0.6758 / 0.6746 | 21 | -72.9% | -100.0% a -28.1% | +5.7% | +1.2% a +10.9% | 65% |
| SP1 | 2320 | 0.9671 | 0.9842 | 0.9673 | 0.9655 | 0.6716 / 0.6692 | 16 | -3.1% | -100.0% a +140.7% | -0.6% | -8.5% a +6.7% | 56% |
| I1 | 2358 | 0.9616 | 0.9784 | 0.9615 | 0.9591 | 0.6830 / 0.6806 | 7 | +200.0% | -21.4% a +485.7% | +12.7% | +4.7% a +22.6% | 86% |
| D1 | 1794 | 0.9836 | 1.0026 | 0.9837 | 0.9802 | 0.6477 / 0.6445 | 16 | -31.2% | -100.0% a +70.3% | +6.9% | +1.4% a +12.0% | 69% |
| F1 | 2027 | 0.9842 | 1.0048 | 0.9843 | 0.9835 | 0.6765 / 0.6710 | 29 | +30.0% | -61.4% a +131.1% | -0.4% | -4.9% a +4.6% | 41% |
| P1 | 1833 | 0.9146 | 0.9345 | 0.9145 | 0.9094 | 0.6734 / 0.6695 | 19 | -33.4% | -100.0% a +52.6% | +3.6% | -4.2% a +11.6% | 47% |
| N1 | 1821 | 0.9404 | 0.9567 | 0.9401 | 0.9346 | 0.6595 / 0.6566 | 46 | +38.9% | -31.5% a +118.7% | +4.6% | -0.2% a +9.7% | 56% |
| **Todas** | **14495** | **0.9603** | **0.9786** | **0.9603** | **0.9570** | **0.6706 / 0.6676** | **154** | **+8.7%** | **-27.7% a +48.4%** | **+3.8%** | **+1.3% a +6.2%** | **56%** |

As colunas "Final" a "O/U 2,5" são log loss (menor é melhor), medido nos mesmos jogos para todas as fontes. O CLV é
calculado contra o fecho da Pinnacle sem margem (Shin): CLV = odd tomada × probabilidade justa − 1. As colunas
`PSC*`/`PC*` existem de 2019/20 até a meio de 2025/26; 152 das 154 apostas têm cobertura.

**Pesos aprendidos**: o modelo recebe peso **a = 0** em todas as 7 épocas validadas fora da amostra (a ≤ 0,03), e o
mercado recebe **b ≈ 1,03**. As odds pré-jogo já contêm tudo o que o Dixon-Coles sabe.

**Comparação com o método proporcional** (a execução anterior, guardada para comparação):
- **Mercado mais preciso**: com Shin, o log loss do mercado pré-jogo passa de 0,9610 para 0,9603 e o do fecho de
  0,9577 para 0,9570. Isto confirma que Shin é o método certo.
- **A combinação deixa de acrescentar**: o ganho que dava face ao mercado pré-jogo (0,9604 contra 0,9610) vinha de
  corrigir o viés que o método proporcional deixava. Com Shin, o mercado já está corrigido (b passa de 1,08 para
  1,03), e a combinação fica igual ao mercado (0,9603).
- **Mais apostas, menos CLV**: as apostas passam de 141 para 154. O CLV contra a Pinnacle passa de +6,2% para +3,8%
  [+1,3%; +6,2%], e a percentagem acima do fecho de 64% para 56%.

**Leitura honesta**:
- A probabilidade final continua **pior do que o fecho em todas as ligas**. Por isso nenhuma liga fica validada e a
  app não mostra `VALOR POTENCIAL`.
- Fica um sinal pequeno de line shopping: CLV de +3,8%, com IC acima de zero. Mas a maior faixa de apostas (odd 5–10,
  89 das 152) não mostra vantagem com nenhum método além do proporcional, e o sinal concentra-se em 17 azarões de odd
  acima de 10. Por liga, Espanha e França ficam negativas.
- Yield de +8,7%, com IC entre −27,7% e +48,4%: sem evidência de rentabilidade.

### Alargado às 21 ligas (24/10/2026)

O mesmo backtest, com as 21 ligas e os pesos da combinação reajustados com todas:

| | Valor |
|---|---|
| Jogos avaliados fora da amostra | 44 329 |
| Log loss 1X2: final / fecho | 0,9978 / 0,9942 |
| Ligas em que o final bate o fecho | **0 de 21** |
| Pesos da combinação | modelo a = 0, mercado b ≈ 1,04 |
| Apostas simuladas | 744 |
| Yield | +4,4% (IC 95% −10,5% a +19,1%) |
| CLV contra o fecho Pinnacle (Shin) | **+2,0% (IC 95% +0,9% a +3,1%)**, 53% acima do fecho |

Com 5 vezes mais apostas, o sinal de line shopping mantém-se pequeno (+2%) mas com IC acima de zero. Continua a não
haver nenhuma liga validada, porque a probabilidade final nunca é tão precisa como o fecho.

### Robustez do CLV a outros métodos de remoção de margem

As apostas são agora escolhidas com Shin. Recalculei o CLV das mesmas 152 apostas com os três métodos, a partir das
linhas originais dos CSVs:

```bash
cd backend && ../.venv/Scripts/python -m app.backtest --clv-robustness
```

| Apostas | N | Proporcional | Power | Shin (o da app) |
|---|---|---|---|---|
| **Todas** | **152** | **+6.6% [+4.1%; +9.1%] · 65%** | **+2.6% [+0.1%; +5.1%] · 52%** | **+3.8% [+1.3%; +6.2%] · 56%** |
| odd ≤ 2 | 4 | +8.0% [-4.0%; +17.8%] · 75% | +9.2% [-3.4%; +19.7%] · 75% | +8.9% [-3.6%; +19.1%] · 75% |
| odd 2–3 | 3 | -5.3% [-15.1%; +7.0%] · 33% | -5.5% [-15.6%; +7.9%] · 33% | -5.4% [-15.4%; +7.7%] · 33% |
| odd 3–5 | 39 | +4.4% [+0.8%; +8.1%] · 62% | +3.4% [-0.4%; +7.3%] · 56% | +3.7% [-0.1%; +7.5%] · 56% |
| odd 5–10 | 89 | +5.0% [+2.1%; +7.7%] · 63% | +0.5% [-2.4%; +3.4%] · 47% | +1.9% [-1.1%; +4.8%] · 53% |
| odd > 10 | 17 | +21.9% [+11.9%; +33.4%] · 88% | +11.5% [+1.8%; +22.6%] · 65% | +14.0% [+4.4%; +25.2%] · 71% |

Cada célula mostra o CLV médio, o IC 95% (bootstrap) e a percentagem de apostas acima do fecho.

Com as apostas escolhidas por Shin, o CLV total fica positivo com os três métodos, mas com power o IC só exclui o
zero por 0,1 pontos. **No máximo, a vantagem de line shopping é pequena (≈ +2,6% a +3,8%)**, assenta em ~150 apostas
em 6 épocas e vem sobretudo de azarões. Mantêm-se estas reservas:
1. **As odds podem não estar disponíveis na prática**: o football-data recolhe as odds pré-jogo uma ou duas vezes por
   semana, sem hora publicada. Uma odd fora do consenso pode ter durado minutos, ou ter limite de aposta baixo. Casas
   como a Bet365 limitam rapidamente contas que ganham assim.
2. **Volume muito baixo**: cerca de 25 apostas por ano nas 7 ligas.
3. **Não é legal a partir de Portugal**: as casas destas apostas (sobretudo Bet365 e Betway) **não têm licença do
   SRIJ**, e não há histórico da Betano/Betclic/Bwin PT nas fontes gratuitas para repetir o teste.

A validação das ligas usa este CLV contra a Pinnacle (Shin) e exige também que a probabilidade final seja tão precisa
como o fecho. Como isso não acontece em nenhuma liga, continua a não haver `VALOR POTENCIAL`.

### Validação de dados das odds

Na primeira execução encontrei apostas em odds claramente erradas (por exemplo, PSG a 5,25 numa casa, quando o
mercado lhe dava cerca de 65%). Acrescentei `ODDS_OUTLIER=0.15`: uma odd mais de 15% acima da odd justa do mercado
fica INCONCLUSIVO. O limite foi fixado por bom senso, não afinado com estes resultados.

Para usar o modelo como ferramenta de análise, as probabilidades finais (≈ mercado calibrado) são as mais fiáveis
que a app consegue dar. Para ter vantagem real seria preciso informação que o mercado pré-jogo ainda não incorporou:
onzes confirmados, lesões de última hora ou movimentos de odds.

## Teste do xG (24/10/2026)

O modelo aceita `XG_WEIGHT`: ataque e defesa ajustam-se a `(1−w)·golos + w·xG` (quase-verosimilhança de Poisson), e a
correção ρ continua a usar os resultados reais. Nos CSVs, **o xG (HxG/AxG) só existe a partir da época 2026/27**, por
isso o teste cobre só essa época. Desenho pré-registado: um único peso candidato (w = 0,5) contra o modelo só com golos,
nos mesmos jogos, com walk-forward semanal e diferença de log loss 1X2 jogo a jogo (IC 95% por bootstrap).

```bash
cd backend && ../.venv/Scripts/python -m app.backtest --compare-xg 0.5 --test-seasons 1
```

| Liga | Jogos | Só golos | Golos + xG | Mercado (fecho) | xG − golos [IC 95%] | xG − mercado [IC 95%] |
|---|---|---|---|---|---|---|
| D1 | 37 | 0.9558 | 0.9478 | 0.9476 | -0.0079 [-0.0166; -0.0004] | +0.0003 [-0.0486; +0.0411] |
| E0 | 57 | 1.0050 | 1.0084 | 0.9871 | +0.0034 [-0.0018; +0.0086] | +0.0213 [-0.0237; +0.0680] |
| F1 | 49 | 0.9994 | 0.9948 | 0.9868 | -0.0046 [-0.0130; +0.0030] | +0.0079 [-0.0410; +0.0542] |
| I1 | 63 | 0.9208 | 0.9184 | 0.9092 | -0.0024 [-0.0158; +0.0089] | +0.0092 [-0.0327; +0.0542] |
| N1 | 64 | 0.9985 | 0.9970 | 0.9845 | -0.0015 [-0.0102; +0.0070] | +0.0126 [-0.0377; +0.0599] |
| P1 | 57 | 0.9255 | 0.9323 | 0.9107 | +0.0068 [+0.0010; +0.0128] | +0.0216 [-0.0121; +0.0576] |
| SP1 | 63 | 0.9251 | 0.9274 | 0.8968 | +0.0023 [-0.0038; +0.0078] | +0.0306 [-0.0113; +0.0687] |
| **Total** | **390** | **0.9604** | **0.9603** | **0.9446** | **-0.0001 [-0.0035; +0.0030]** | **+0.0158 [-0.0009; +0.0326]** |

**Conclusão**: no total, sem melhoria mensurável (a diferença é zero dentro do IC). O resultado melhor na Bundesliga
e pior em Portugal é o que se espera do acaso ao comparar 7 ligas, e não serve para escolher pesos por liga. O mercado
continua mais preciso. Por isso `XG_WEIGHT` fica em 0.

Porque é que o teste é fraco: no início de 2026/27 quase todo o treino é de épocas sem xG, e o decaimento temporal dá
peso às épocas anteriores, por isso a parte de xG no alvo de treino é pequena. Vale a pena repetir no fim da época, ou
com uma fonte de xG histórica autorizada.

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/health` | estado, Odds API ativa e quota |
| GET | `/api/leagues` | ligas configuradas |
| GET | `/api/fixtures?league=P1` | próximos jogos com resumo e categoria |
| GET | `/api/match?key=P1\|2026-10-24\|Casa\|Fora` | relatório completo A–E |
| GET | `/api/history` | previsões registadas e resultado real |
| GET | `/api/performance` | métricas das previsões já resolvidas (log loss, Brier, calibração, ROI simulado) |
| GET | `/api/alerts` | variações de odds acima de `ODDS_MOVE_ALERT` |
| GET | `/api/backtests` | resultados guardados |
| POST | `/api/admin/backtest` | corre o backtest de todas as ligas, ≈10 min (cabeçalho `X-Admin-Token`) |
| POST | `/api/admin/refresh` | expira as caches (cabeçalho `X-Admin-Token`) |

Documentação interativa: http://localhost:8000/docs

## Limitações conhecidas

- Sem lesões, suspensões, onzes, meteorologia nem notícias: é a principal razão para o mercado ser mais preciso.
- Só jogos do campeonato: taças e competições europeias não contam para a força nem para o descanso.
- Equipas recém-promovidas ficam sem previsão até terem 10 jogos na liga.
- As odds do football-data.co.uk são recolhidas uma ou duas vezes por semana e a hora exata não é publicada, por isso
  os alertas de variação só são úteis com a Odds API.
- O cálculo de valor usa a melhor odd entre operadores, o que pressupõe ter conta em todos.

## Próximos passos sugeridos

1. Integrar uma fonte de lesões e onzes. Repetir o teste do xG no fim da época 2026/27 (`--compare-xg 0.5`).
2. Começar a guardar odds das casas SRIJ (odds-api.io) para medir o CLV em casas legais em Portugal.
3. Cantos e cartões (os dados já existem nos CSVs).
4. Outras modalidades, cada uma com um módulo próprio em `models/`.
5. PostgreSQL, Redis e frontend Next.js quando houver vários utilizadores.
