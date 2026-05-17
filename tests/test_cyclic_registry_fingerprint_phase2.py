from unittest.mock import patch

import numpy as np
from Orange.data import Domain, StringVariable, Table
from Orange.widgets.tests.base import WidgetTest

from chem_inf_widgets.chemcore.descriptors.cyclic_registry_fingerprint import (
    BIT_SECTIONS,
    DEFAULT_N_BITS,
    RegistryEntry,
    _bit_names,
    compute_cyclic_registry_fingerprints_from_smiles,
    load_registry_entries,
)
from chem_inf_widgets.widgets.ow_cyclic_registry_fingerprint import (
    OWCyclicRegistryFingerprint,
)


def test_registry_loads_entries():
    version, entries = load_registry_entries()
    assert version
    assert len(entries) > 0
    assert entries[0].entry_id
    assert entries[0].smarts


def test_bit_layout_is_4096():
    assert DEFAULT_N_BITS == 4096
    assert BIT_SECTIONS["morgan"] == (0, 2048)
    assert BIT_SECTIONS["heterocycle_registry"] == (2048, 3072)
    assert BIT_SECTIONS["reserved"] == (3968, 4096)


def test_compute_small_panel():
    res = compute_cyclic_registry_fingerprints_from_smiles(
        ["c1ccncc1", "C1CCCCC1", "not_smiles"],
        max_registry_entries=300,
    )
    assert res.X.shape[1] == 4096
    assert len(res.valid_indices) == 2
    assert len(res.failed_indices) == 1
    assert len(res.bit_names) == 4096
    assert res.registry_version


def test_bit_names_surface_hash_collisions():
    entries = [
        RegistryEntry(entry_id="E1", name="One", smarts="c1ccccc1", ring_count=1),
        RegistryEntry(entry_id="E2", name="Two", smarts="n1ccccc1", ring_count=1),
    ]
    with patch(
        "chem_inf_widgets.chemcore.descriptors.cyclic_registry_fingerprint._bit_for_entry",
        return_value=2050,
    ):
        names = _bit_names(entries, include_morgan=False)
    assert "collision2" in names[2050]
    assert "E1" in names[2050]
    assert "E2" in names[2050]


class TestOWCyclicRegistryFingerprint(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCyclicRegistryFingerprint)

    @staticmethod
    def _make_table() -> Table:
        domain = Domain(
            [],
            metas=[
                StringVariable("SMILES"),
                StringVariable("name"),
                StringVariable("note"),
            ],
        )
        metas = np.array(
            [
                ["c1ccncc1", "mol-a", "keep-a"],
                ["not_smiles", "mol-b", "drop-b"],
                ["C1CCCCC1", "mol-c", "keep-c"],
            ],
            dtype=object,
        )
        return Table.from_numpy(domain, X=np.empty((3, 0)), metas=metas)

    def test_molecules_output_stays_aligned_with_valid_rows(self):
        table = self._make_table()
        res = compute_cyclic_registry_fingerprints_from_smiles(
            ["c1ccncc1", "not_smiles", "C1CCCCC1"],
            max_registry_entries=300,
        )
        mols = self.widget._build_molecules_output(res, table, None)
        assert mols is not None
        assert len(mols) == res.X.shape[0] == 2
        assert [m.name for m in mols] == ["mol-a", "mol-c"]
        assert [m.props.get("note") for m in mols] == ["keep-a", "keep-c"]
        assert [m.props.get("source_row_index") for m in mols] == [0, 2]
        assert [m.props.get("SMILES") for m in mols] == list(res.smiles)
        for mol in mols:
            assert mol.props.get("fp", {}).get("type") == "cyclic_registry_4096"

    def test_match_table_includes_source_row_index(self):
        table = self._make_table()
        res = compute_cyclic_registry_fingerprints_from_smiles(
            ["c1ccncc1", "not_smiles", "C1CCCCC1"],
            max_registry_entries=300,
        )
        match_table = self.widget._build_match_table(res)
        assert match_table is not None
        attr_names = [var.name for var in match_table.domain.attributes]
        meta_names = [var.name for var in match_table.domain.metas]
        assert attr_names == ["valid_row_index", "source_row_index", "bit", "match_count"]
        assert meta_names[:2] == ["SMILES", "inchikey"]
        source_rows = set(int(v) for v in match_table.get_column("source_row_index"))
        assert source_rows.issubset({0, 2})
        assert all(str(v).strip() for v in match_table.get_column("inchikey"))

    def test_fingerprint_table_has_modeling_identifiers_first(self):
        table = self._make_table()
        res = compute_cyclic_registry_fingerprints_from_smiles(
            ["c1ccncc1", "not_smiles", "C1CCCCC1"],
            max_registry_entries=300,
        )
        fp_table = self.widget._build_fingerprint_table(res, table)
        meta_names = [var.name for var in fp_table.domain.metas]
        assert meta_names[:2] == ["SMILES", "inchikey"]
        assert list(fp_table.get_column("SMILES")) == list(res.smiles)
        assert all(str(v).strip() for v in fp_table.get_column("inchikey"))
