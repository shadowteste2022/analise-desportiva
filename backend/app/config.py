from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Optional external services. Without them the app still works on football-data.co.uk.
    odds_api_key: str | None = None
    odds_api_min_remaining: int = 20  # stop calling The Odds API below this quota
    odds_api_monthly_budget: int = 500  # credits per billing period (free plan = 500)
    odds_api_reset_day: int = Field(1, ge=1, le=28)  # day of the month the plan renews
    # odds-api.io: the only feed found with SRIJ-licensed books (Betano PT, Betclic PT, Bwin PT)
    odds_api_io_key: str | None = None
    odds_api_io_bookmakers: str = "Betano PT,Betclic PT,Bwin PT"
    # OddsPapi: free plan (250 requests/month) with SRIJ-licensed Betano PT
    # API-Football: world fixtures and results (free plan: 100 requests/day)
    api_football_key: str | None = None
    oddspapi_key: str | None = None
    oddspapi_monthly_budget: int = 250
    oddspapi_reset_day: int = Field(1, ge=1, le=28)
    admin_token: str | None = None  # required for POST /api/admin/*; admin disabled if unset

    # Other sports via The Odds API (market consensus only). "*" = every in-season sport and competition the API
    # lists (men's and women's), minus outright-only markets. Otherwise a comma list; entries ending in "_" are
    # prefixes (tennis keys change with each tournament).
    other_sports: str = "*"
    data_dir: Path = ROOT / "data"
    leagues: str = ("P1,E0,SP1,I1,D1,F1,N1,B1,T1,G1,SC0,E1,SP2,I2,D2,F2,E2,E3,EC,SC1,SC2,"
                    "BRA,ARG,USA,MEX,JPN,CHN,AUT,SWZ,DNK,NOR,SWE,FIN,POL,ROU,IRL,RUS")
    seasons_back: int = 3

    results_ttl_hours: float = 12
    fixtures_ttl_hours: float = 1
    odds_max_age_hours: float = 48  # older odds -> "SEM DADOS SUFICIENTES"
    # The Odds API odds are reused this long: forecasts barely move hour to hour, and the free credits then reach
    # many more sports and leagues
    odds_ttl_hours: float = 12
    refresh_minutes: int = 0  # background refresh of fixtures/odds; 0 = off

    # Model / decision parameters
    time_decay_xi: float = 0.0019  # per day, Dixon & Coles (1997) ~ half-life of one year
    xg_weight: float = 0.0  # share of xG in the fitting target (0 = goals only); see README backtest
    min_team_matches: int = 10
    ev_margin: float = 0.03  # minimum EV per unit for "VALOR POTENCIAL"
    # Only SRIJ-licensed bookmakers (legal in Portugal) for prices, best bets and the odds tables shown to visitors.
    # Unlicensed books still feed the no-vig market reference behind the probabilities.
    pt_only: bool = False
    devig_method: str = "shin"  # proportional | power | shin; Shin corrects the favourite-longshot bias
    max_market_divergence: float = 0.12  # |model - no-vig market| above this -> inconclusive
    odds_outlier: float = 0.15  # best odds this far above the market's fair odds -> data error/stale, inconclusive
    odds_move_alert: float = 0.08  # relative odds move that triggers an alert

    @field_validator("odds_api_key", "odds_api_io_key", "oddspapi_key", "api_football_key", "admin_token", mode="before")
    @classmethod
    def _clean_secret(cls, v):
        """Keys pasted into GitHub secrets or .env often carry a newline, spaces or quotes, which the APIs reject."""
        return v.strip().strip("'\"").strip() or None if isinstance(v, str) else v

    @property
    def league_list(self) -> list[str]:
        return [x.strip() for x in self.leagues.split(",") if x.strip()]


settings = Settings()
