from __future__ import annotations

from typing import List, Optional

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QCheckBox, QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from Orange.data import ContinuousVariable, Domain, StringVariable, Table
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget

from chem_inf_widgets.chemcore.services.reaction_enumerator_service import enumerate_reaction_products
from chem_inf_widgets.widgets.ui_helpers import (
    format_required_inputs_status,
    format_result_count_status,
    format_waiting_status,
)


def _guess_smiles_column(data: Table) -> Optional[StringVariable]:
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    for variable in variables:
        if isinstance(variable, StringVariable) and variable.name.strip().lower() in {"smiles", "canonical_smiles", "smile"}:
            return variable
    return next((variable for variable in variables if isinstance(variable, StringVariable)), None)


def _guess_smirks_column(data: Table) -> Optional[StringVariable]:
    variables = list(data.domain.metas) + list(data.domain.attributes) + list(data.domain.class_vars)
    for variable in variables:
        if isinstance(variable, StringVariable) and variable.name.strip().lower() in {"smirks", "reaction", "smarts"}:
            return variable
    return next((variable for variable in variables if isinstance(variable, StringVariable)), None)


def _table_column_as_strings(data: Table, variable: Optional[StringVariable]) -> List[str]:
    if variable is None:
        return []
    return ["" if value is None else str(value).strip() for value in data.get_column(variable)]


class OWReactionEnumerator(OWWidget):
    name = "Reaction Enumerator"
    description = "Enumerate virtual products from one or more reactant sets and SMIRKS rules."
    icon = "icons/reactions/owreactionenumeratorwidget.svg"
    priority = 153

    class Inputs:
        reactants_a = Input("Reactants A", Table)
        reactants_b = Input("Reactants B", Table)
        reactants_c = Input("Reactants C", Table)
        reactions = Input("Reactions", Table)

    class Outputs:
        products = Output("Products", Table)
        summary = Output("Enumeration Summary", Table)

    max_products: int = Setting(500)
    auto_run: bool = Setting(False)

    def __init__(self) -> None:
        super().__init__()
        self.reactants_a: Optional[Table] = None
        self.reactants_b: Optional[Table] = None
        self.reactants_c: Optional[Table] = None
        self.reactions: Optional[Table] = None
        self.mainArea.hide()

        root = QWidget(self.controlArea)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        self.controlArea.layout().addWidget(root)

        self.status_label = QLabel(format_waiting_status("reactants and reactions"))
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setLabelAlignment(Qt.AlignLeft)
        self.max_products_spin = QSpinBox()
        self.max_products_spin.setRange(1, 100000)
        self.max_products_spin.setValue(int(self.max_products))
        self.max_products_spin.valueChanged.connect(self._on_max_products_changed)
        form.addRow("Max products:", self.max_products_spin)
        layout.addWidget(form_widget)

        self.auto_run_check = QCheckBox("Auto-run")
        self.auto_run_check.setChecked(bool(self.auto_run))
        self.auto_run_check.toggled.connect(self._on_auto_run_toggled)
        layout.addWidget(self.auto_run_check)

        run_button = QPushButton("Enumerate products")
        run_button.clicked.connect(self.commit)
        layout.addWidget(run_button)
        layout.addStretch(1)

    @Inputs.reactants_a
    def set_reactants_a(self, data: Optional[Table]) -> None:
        self.reactants_a = data
        self._maybe_autorun()

    @Inputs.reactants_b
    def set_reactants_b(self, data: Optional[Table]) -> None:
        self.reactants_b = data
        self._maybe_autorun()

    @Inputs.reactants_c
    def set_reactants_c(self, data: Optional[Table]) -> None:
        self.reactants_c = data
        self._maybe_autorun()

    @Inputs.reactions
    def set_reactions(self, data: Optional[Table]) -> None:
        self.reactions = data
        self._maybe_autorun()

    def _on_max_products_changed(self, value: int) -> None:
        self.max_products = int(value)
        self._maybe_autorun()

    def _on_auto_run_toggled(self, checked: bool) -> None:
        self.auto_run = bool(checked)
        self._maybe_autorun()

    def _maybe_autorun(self) -> None:
        if bool(self.auto_run) and self.reactants_a is not None and self.reactions is not None:
            self.commit()

    def commit(self) -> None:
        if self.reactants_a is None or self.reactions is None:
            self.status_label.setText(format_required_inputs_status("Reactants A", "Reactions"))
            self.Outputs.products.send(None)
            self.Outputs.summary.send(None)
            return

        reactant_sets = []
        for table in (self.reactants_a, self.reactants_b, self.reactants_c):
            if table is None:
                continue
            reactant_sets.append(_table_column_as_strings(table, _guess_smiles_column(table)))

        smirks_var = _guess_smirks_column(self.reactions)
        name_var = next(
            (
                variable
                for variable in list(self.reactions.domain.metas) + list(self.reactions.domain.attributes) + list(self.reactions.domain.class_vars)
                if isinstance(variable, StringVariable) and variable.name.strip().lower() in {"name", "title", "rule"}
            ),
            None,
        )
        smirks_values = _table_column_as_strings(self.reactions, smirks_var)
        name_values = _table_column_as_strings(self.reactions, name_var)
        reaction_rows = [
            (name_values[index] if index < len(name_values) and name_values[index] else f"Rule {index + 1}", smirks)
            for index, smirks in enumerate(smirks_values)
            if smirks
        ]

        products = enumerate_reaction_products(reactant_sets, reaction_rows, max_products=int(self.max_products))
        self.Outputs.products.send(self._products_table(products))
        self.Outputs.summary.send(self._summary_table(products))
        self.status_label.setText(format_result_count_status(len(products), item_label="products", prefix="Enumerated"))

    def _products_table(self, products) -> Table:
        metas = [
            StringVariable("Rule Name"),
            StringVariable("Rule SMIRKS"),
            StringVariable("Reactants"),
            StringVariable("Product SMILES"),
        ]
        domain = Domain([], metas=metas)
        rows = [
            [product.rule_name, product.rule_smirks, " + ".join(product.reactants), product.product_smiles]
            for product in products
        ]
        metas_arr = np.array(rows, dtype=object) if rows else np.zeros((0, 4), dtype=object)
        return Table.from_numpy(domain, X=np.zeros((len(rows), 0), dtype=float), metas=metas_arr)

    def _summary_table(self, products) -> Table:
        count_var = ContinuousVariable("Count")
        rule_var = StringVariable("Rule Name")
        counts = {}
        for product in products:
            counts[product.rule_name] = counts.get(product.rule_name, 0) + 1
        rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        domain = Domain([count_var], metas=[rule_var])
        X = np.array([[count] for _rule, count in rows], dtype=float) if rows else np.zeros((0, 1), dtype=float)
        metas = np.array([[rule] for rule, _count in rows], dtype=object) if rows else np.zeros((0, 1), dtype=object)
        return Table.from_numpy(domain, X=X, metas=metas)
