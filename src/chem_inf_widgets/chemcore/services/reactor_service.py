from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from chem_inf_widgets.chemcore.services.rdkit_safe import safe_canonical_smiles, safe_mol_from_smiles


@dataclass
class ReactionRule:
    name: str
    smirks: str
    rxn: rdChemReactions.ChemicalReaction
    weight: float = 1.0

    @property
    def n_reactants(self) -> int:
        return self.rxn.GetNumReactantTemplates()

    @staticmethod
    def from_row(name: str, smirks: str, weight: Optional[float] = None) -> "ReactionRule":
        rxn = rdChemReactions.ReactionFromSmarts(smirks, useSmiles=True)
        if rxn is None:
            raise ValueError(f"Invalid SMIRKS/SMILES reaction: {smirks}")
        rxn.Initialize()
        parsed_weight = 1.0 if weight is None else max(0.0, float(weight))
        return ReactionRule(name=name or smirks, smirks=smirks, rxn=rxn, weight=parsed_weight)


def coerce_seed(seed_text: str, default: int = 0) -> int:
    try:
        return int((seed_text or "").strip())
    except ValueError:
        return default


def build_preview_text(lines: Sequence[str], max_lines: int) -> str:
    visible_lines = list(lines[:max_lines])
    if len(lines) > max_lines:
        visible_lines.append(f"... ({len(lines) - max_lines} more lines)")
    return "\n".join(visible_lines)


class ReactorEngine:
    def __init__(
        self,
        smiles: List[str],
        rules: List[ReactionRule],
        seed: Optional[int] = None,
        sanitize_products: bool = True,
        allow_self_react: bool = False,
        unique_products: bool = True,
        consume_reactants: bool = False,
        expand_pool: bool = False,
        max_pool_size: int = 0,
        pool_policy: str = "fifo",
        per_rule_max_trials: int = 2000,
    ):
        self.rng = random.Random(seed)
        dedup: Dict[str, ReactionRule] = {}
        for rule in rules:
            key = rule.smirks.strip()
            if key in dedup:
                dedup[key].weight += rule.weight
            else:
                dedup[key] = rule
        self.rules = list(dedup.values())

        self.allow_self = allow_self_react
        self.sanitize = sanitize_products
        self.unique_products = unique_products
        self.consume_reactants = consume_reactants
        self.expand_pool = expand_pool
        self.max_pool_size = max(0, int(max_pool_size))
        self.pool_policy = (pool_policy or "fifo").lower()
        self.per_rule_max_trials = max(100, int(per_rule_max_trials))

        self.mol_pool: List[Chem.Mol] = []
        self.smiles_pool: List[str] = []
        for smiles_value in smiles:
            parsed = safe_mol_from_smiles(smiles_value, remove_hs=False)
            mol = parsed.mol
            if mol is None:
                continue
            canonical_smiles = safe_canonical_smiles(mol, remove_hs=False)
            if not canonical_smiles:
                continue
            self.mol_pool.append(mol)
            self.smiles_pool.append(canonical_smiles)
        self._enforce_pool_cap()

        self._seen_products: set[str] = set()
        self.last_preview: List[str] = []

    def _applicable_indices(self, rule: ReactionRule) -> List[List[int]]:
        idxs_per_template: List[List[int]] = []
        for template_index in range(rule.n_reactants):
            template = rule.rxn.GetReactantTemplate(template_index)
            matches: List[int] = []
            for mol_index, mol in enumerate(self.mol_pool):
                if mol.HasSubstructMatch(template):
                    matches.append(mol_index)
            idxs_per_template.append(matches)
        return idxs_per_template

    def _choose_combos(
        self, idxs_per_template: List[List[int]], max_combos: int
    ) -> List[Tuple[int, ...]]:
        if not idxs_per_template or any(len(indices) == 0 for indices in idxs_per_template):
            return []
        sizes = [len(indices) for indices in idxs_per_template]
        total = 1
        for size in sizes:
            total *= size
        if total <= 2000:
            combos = list(itertools.product(*idxs_per_template))
            if not self.allow_self:
                combos = [combo for combo in combos if len(set(combo)) == len(combo)]
            self.rng.shuffle(combos)
            return combos[:max_combos]

        combos: List[Tuple[int, ...]] = []
        tries = 0
        max_tries = max(self.per_rule_max_trials, max_combos * 10)
        while len(combos) < max_combos and tries < max_tries:
            pick = tuple(self.rng.choice(indices) for indices in idxs_per_template)
            if not self.allow_self and len(set(pick)) != len(pick):
                tries += 1
                continue
            if pick not in combos:
                combos.append(pick)
            tries += 1
        return combos

    def _sanitize_and_smiles(self, mol: Chem.Mol) -> Optional[str]:
        if self.sanitize:
            try:
                Chem.SanitizeMol(mol)
            except (RuntimeError, ValueError):
                return None
        smiles = safe_canonical_smiles(mol, remove_hs=False)
        return smiles or None

    def _weighted_rule_order(self) -> List[ReactionRule]:
        if not self.rules:
            return []
        weights = [max(0.0, rule.weight) for rule in self.rules]
        if sum(weights) <= 0:
            order = list(self.rules)
            self.rng.shuffle(order)
            return order
        picks = self.rng.choices(range(len(self.rules)), weights=weights, k=len(self.rules) * 3)
        seen = set()
        order_indices: List[int] = []
        for index in picks:
            if index not in seen:
                seen.add(index)
                order_indices.append(index)
            if len(order_indices) == len(self.rules):
                break
        for index in range(len(self.rules)):
            if index not in seen:
                order_indices.append(index)
        return [self.rules[index] for index in order_indices]

    def _enforce_pool_cap(self) -> None:
        if self.max_pool_size <= 0:
            return
        pool_size = len(self.smiles_pool)
        if pool_size <= self.max_pool_size:
            return
        if self.pool_policy == "random":
            keep_indices = sorted(self.rng.sample(range(pool_size), self.max_pool_size))
        elif self.pool_policy == "lifo":
            keep_indices = list(range(self.max_pool_size))
        else:
            keep_indices = list(range(max(0, pool_size - self.max_pool_size), pool_size))
        self.mol_pool = [self.mol_pool[index] for index in keep_indices]
        self.smiles_pool = [self.smiles_pool[index] for index in keep_indices]

    def _rxn_to_string(self, rxn: rdChemReactions.ChemicalReaction) -> str:
        try:
            params = rdChemReactions.SmilesWriteParams()
            params.canonical = True
            return rdChemReactions.ReactionToSmiles(rxn, params)
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            return rdChemReactions.ReactionToSmiles(rxn)
        except RuntimeError:
            pass
        try:
            return rdChemReactions.ReactionToSmarts(rxn)
        except RuntimeError:
            return ""

    def _append_products_to_pool(self, product_smiles: Sequence[str]) -> None:
        for smiles_value in product_smiles:
            parsed = safe_mol_from_smiles(smiles_value, remove_hs=False)
            mol = parsed.mol
            if mol is None:
                continue
            canonical_smiles = safe_canonical_smiles(mol, remove_hs=False)
            if not canonical_smiles:
                continue
            if canonical_smiles in self.smiles_pool:
                continue
            self.mol_pool.append(mol)
            self.smiles_pool.append(canonical_smiles)
        self._enforce_pool_cap()

    def step(
        self, draws_per_step: int = 5, max_products_per_draw: int = 4
    ) -> List[Dict[str, Any]]:
        self.last_preview = []
        records: List[Dict[str, Any]] = []
        if not self.rules or not self.mol_pool:
            return records

        used_indices: set[int] = set()
        products_to_add: List[str] = []

        chosen_rule: Optional[ReactionRule] = None
        combos: List[Tuple[int, ...]] = []
        for rule in self._weighted_rule_order():
            idxs = self._applicable_indices(rule)
            candidate_combos = self._choose_combos(idxs, draws_per_step)
            if candidate_combos:
                chosen_rule = rule
                combos = candidate_combos
                break
        if chosen_rule is None:
            return records

        mapped_str = self._rxn_to_string(chosen_rule.rxn)
        self.last_preview.append(
            f"Picked rule: {chosen_rule.name}  [weight={chosen_rule.weight:.3g}]\nSMIRKS: {chosen_rule.smirks}"
        )

        for combo in combos:
            if self.consume_reactants and any(index in used_indices for index in combo):
                continue
            reactant_mols = tuple(self.mol_pool[index] for index in combo)
            reactant_smiles = [self.smiles_pool[index] for index in combo]
            try:
                outcomes = chosen_rule.rxn.RunReactants(reactant_mols)
            except RuntimeError:
                continue
            self.last_preview.append("— Match: " + " + ".join(reactant_smiles))

            count = 0
            for prod_tuple in outcomes:
                parts: List[str] = []
                for product_mol in prod_tuple:
                    smiles_value = self._sanitize_and_smiles(product_mol)
                    if smiles_value is None:
                        parts = []
                        break
                    parts.append(smiles_value)
                if not parts:
                    continue
                parts_sorted = sorted(parts)
                product_smiles = ".".join(parts_sorted)
                if self.unique_products and product_smiles in self._seen_products:
                    continue
                self._seen_products.add(product_smiles)
                records.append(
                    {
                        "product_smiles": product_smiles,
                        "reactant_smiles": " + ".join(reactant_smiles),
                        "reaction_name": chosen_rule.name,
                        "smirks": chosen_rule.smirks,
                        "rxn_mapped": mapped_str,
                    }
                )
                self.last_preview.append("   → Products: " + product_smiles)
                if self.expand_pool:
                    products_to_add.extend(parts_sorted)
                count += 1
                if count >= max_products_per_draw:
                    break
            if self.consume_reactants and count > 0:
                used_indices.update(combo)

        if self.consume_reactants and used_indices:
            for index in sorted(used_indices, reverse=True):
                self.mol_pool.pop(index)
                self.smiles_pool.pop(index)
        if self.expand_pool and products_to_add:
            self._append_products_to_pool(products_to_add)

        return records

    def run(
        self, n_steps: int = 1, draws_per_step: int = 5, max_products_per_draw: int = 4
    ) -> List[Dict[str, Any]]:
        all_records: List[Dict[str, Any]] = []
        for _ in range(max(1, n_steps)):
            all_records.extend(
                self.step(
                    draws_per_step=draws_per_step,
                    max_products_per_draw=max_products_per_draw,
                )
            )
        return all_records
