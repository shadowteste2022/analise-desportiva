"""Which bookmakers hold a sports-betting licence from SRIJ (Portugal's regulator), the only ones allowed to take bets
from Portugal. List checked on 24/10/2026 at srij.turismodeportugal.pt; licences change, so re-check it.

Odds feeds name the Portuguese books with a " PT" suffix ("Betano PT", "Bwin PT"); the rest are matched by brand.
Live feeds append " · live" to the bookmaker name.
"""
BRANDS = ("placard", "solverde", "esc online", "estoril sol", "casino portugal", "lebull", "goldenpark pt",
          "retabet pt")


def licensed(book: str) -> bool:
    name = (book or "").split(" · ")[0].strip()
    return name.endswith(" PT") or any(b in name.lower() for b in BRANDS)
