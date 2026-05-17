from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

from rdkit.Chem import rdChemReactions


def pick_preferred_column(
    column_names: Sequence[str],
    preferred_names: Iterable[str],
    fallback: str = "",
) -> str:
    for preferred_name in preferred_names:
        for column_name in column_names:
            if column_name.lower() == preferred_name.lower():
                return column_name
    return column_names[0] if column_names else fallback


def compose_reaction_string(
    reactants_text: Optional[str],
    products_text: Optional[str],
) -> Optional[str]:
    reactants_raw = (reactants_text or "").strip()
    products_raw = (products_text or "").strip()
    if not reactants_raw or not products_raw or reactants_raw == "?" or products_raw == "?":
        return None

    reactants = [
        item.strip()
        for item in reactants_raw.replace(".", "+").split("+")
        if item.strip() and item.strip() != "?"
    ]
    products = [
        item.strip()
        for item in products_raw.split(".")
        if item.strip() and item.strip() != "?"
    ]
    if not reactants or not products:
        return None
    return ".".join(reactants) + ">>" + ".".join(products)


def parse_reaction_string(rxn_str: str) -> Optional[rdChemReactions.ChemicalReaction]:
    if not rxn_str:
        return None
    try:
        reaction = rdChemReactions.ReactionFromSmarts(rxn_str, useSmiles=True)
        if reaction is None:
            raise ValueError("No reaction parsed from SMILES-mode input")
        reaction.Initialize()
        return reaction
    except (RuntimeError, ValueError):
        pass

    try:
        reaction = rdChemReactions.ReactionFromSmarts(rxn_str, useSmiles=False)
        if reaction is None:
            return None
        reaction.Initialize()
        return reaction
    except RuntimeError:
        return None


def safe_slug(text: str) -> str:
    slug = re.sub(r"\s+", "_", text)
    slug = re.sub(r"[^A-Za-z0-9_.-]", "", slug)
    return slug[:80] or "rxn"


def build_export_name(prefix: str, row_index: int, caption: Optional[str]) -> str:
    normalized_prefix = prefix or "rxn_"
    normalized_caption = caption or "rxn"
    return f"{normalized_prefix}{row_index:04d}_{safe_slug(normalized_caption)}"
