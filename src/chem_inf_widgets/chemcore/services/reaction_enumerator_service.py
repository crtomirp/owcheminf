from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from rdkit import Chem

from chem_inf_widgets.chemcore.services.reactor_service import ReactionRule
from chem_inf_widgets.chemcore.services.rdkit_safe import (
    safe_canonical_smiles,
    safe_mol_from_smiles,
)


@dataclass(frozen=True)
class EnumeratedProduct:
    rule_name: str
    rule_smirks: str
    reactants: tuple[str, ...]
    product_smiles: str


def enumerate_reaction_products(
    reactant_sets: Sequence[Sequence[str]],
    reaction_rows: Sequence[tuple[str, str]],
    *,
    max_products: int = 1000,
    unique_products: bool = True,
) -> list[EnumeratedProduct]:
    clean_sets = [[(value or "").strip() for value in values if (value or "").strip()] for values in reactant_sets]
    clean_sets = [values for values in clean_sets if values]
    if not clean_sets:
        return []

    rules = [ReactionRule.from_row(name, smirks) for name, smirks in reaction_rows if (smirks or "").strip()]
    if not rules:
        return []

    products: list[EnumeratedProduct] = []
    seen = set()
    for rule in rules:
        n_reactants = rule.n_reactants
        if n_reactants == 0 or n_reactants > len(clean_sets):
            continue

        pools = clean_sets[:n_reactants]
        stack = [([], 0)]
        while stack:
            prefix, level = stack.pop()
            if level == len(pools):
                mols = []
                for smiles in prefix:
                    mol = safe_mol_from_smiles(smiles, sanitize=True, remove_hs=True).mol
                    if mol is None:
                        mols = []
                        break
                    mols.append(mol)
                if not mols:
                    continue
                try:
                    outcomes = rule.rxn.RunReactants(tuple(mols))
                except Exception:
                    continue
                for outcome in outcomes:
                    product_parts = []
                    for mol in outcome:
                        try:
                            Chem.SanitizeMol(mol)
                            product_smi = safe_canonical_smiles(mol, remove_hs=False)
                            if product_smi:
                                product_parts.append(product_smi)
                        except Exception:
                            continue
                    if not product_parts:
                        continue
                    product_smiles = ".".join(product_parts)
                    dedup_key = (rule.smirks, product_smiles)
                    if unique_products and dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    products.append(
                        EnumeratedProduct(
                            rule_name=rule.name,
                            rule_smirks=rule.smirks,
                            reactants=tuple(prefix),
                            product_smiles=product_smiles,
                        )
                    )
                    if len(products) >= max_products:
                        return products
                continue

            for smiles in pools[level]:
                stack.append((prefix + [smiles], level + 1))

    return products
