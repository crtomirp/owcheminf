from chem_inf_widgets.chemcore.descriptors.cyclic_registry_fingerprint import (
    RegistryEntry,
    load_registry_entries,
)
from chem_inf_widgets.chemcore.descriptors.cyclic_registry_validation import (
    analyze_cyclic_registry,
    collision_rows,
    format_registry_report,
    report_to_json,
)


def test_phase21_packaged_registry_metadata_report_runs_without_smarts_compile():
    report = analyze_cyclic_registry(limit=250, compile_smarts=False)
    assert report.total_entries >= report.selected_entries > 0
    assert report.registry_version
    assert any(s.section == "heterocycle_registry" for s in report.section_stats)
    assert "bit_sections" in report.metadata
    assert report.error_count >= 0


def test_phase21_custom_duplicate_and_invalid_smarts_are_reported():
    entries = [
        RegistryEntry(entry_id="DUP", name="benzene A", smarts="c1ccccc1", ring_count=1),
        RegistryEntry(entry_id="DUP", name="benzene B", smarts="c1ccccc1", ring_count=1),
        RegistryEntry(entry_id="BAD", name="bad smarts", smarts="[", group="functional_group"),
    ]
    report = analyze_cyclic_registry(entries, compile_smarts=True)
    codes = {issue.code for issue in report.issues}
    assert "duplicate_entry_id" in codes
    assert "invalid_smarts" in codes or "smarts_exception" in codes
    assert report.error_count >= 2


def test_phase21_collision_rows_are_flattened_for_csv_export():
    entries = [
        RegistryEntry(entry_id=f"E{i}", name=f"entry {i}", smarts="c1ccccc1", ring_count=1)
        for i in range(6)
    ]
    # Identical entries are not a realistic registry design, but this gives a
    # deterministic collision-independent test of CSV flattening and formatting.
    report = analyze_cyclic_registry(entries, compile_smarts=False)
    text = format_registry_report(report)
    assert "Cyclic Registry Validation Report" in text
    js = report_to_json(report)
    assert '"section_stats"' in js
    rows = collision_rows(report)
    assert isinstance(rows, list)


def test_phase21_full_packaged_registry_has_expected_size_without_smarts_compile():
    _, entries = load_registry_entries()
    report = analyze_cyclic_registry(entries, compile_smarts=False)
    assert report.selected_entries == len(entries)
    assert report.selected_entries >= 2000
    assert sum(s.entries for s in report.section_stats) == report.selected_entries
