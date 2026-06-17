"""Effect registry.

``main.py`` imports :func:`build_effects` and otherwise knows nothing about
which fruits exist. To add a fruit: write its module, then add one line here.
The order is the order shown on the button bar; "Off" stays first so the app
opens on the raw feed.
"""

from __future__ import annotations

from typing import List

from effects.base import BaseEffect
from effects.gum_gum import GumGum
from effects.hie_hie import HieHie
from effects.mera_mera import MeraMera
from effects.moku_moku import MokuMoku
from effects.none_effect import NoEffect
from effects.suna_suna import SunaSuna


def build_effects() -> List[BaseEffect]:
    """Instantiate every effect, in button-bar order."""
    return [
        NoEffect(),
        GumGum(),
        MeraMera(),
        HieHie(),
        SunaSuna(),
        MokuMoku(),
    ]


__all__ = ["build_effects", "BaseEffect"]
