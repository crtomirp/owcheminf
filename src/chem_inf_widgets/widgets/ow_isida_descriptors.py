from __future__ import annotations

import itertools
import logging
from collections import Counter
from typing import List, Optional, Dict, Tuple

import numpy as np
from AnyQt.QtWidgets import (
    QLabel, QProgressBar, QComboBox, QCheckBox, QHBoxLayout
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from chem_inf_widgets.widgets.ui_helpers import clear_widget_messages, set_widget_error, set_widget_warning

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from chem_inf_widgets.chemcore.mol import ChemMol
except Exception:
    ChemMol = object
from chem_inf_widgets.chemcore.services.rdkit_safe import safe_mol_from_smiles


logger = logging.getLogger(__name__)


class IsidaWorker(QThread):
    """Calculates ISIDA-like descriptors in a background thread."""
    progress = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, mols, topology, min_len, max_len, min_support, include_atoms, include_bonds):
        super().__init__()
        self.mols = mols
        self.topology = topology  # "I", "II", "III"
        self.min_len = min_len
        self.max_len = max_len
        self.min_support = min_support
        self.include_atoms = include_atoms
        self.include_bonds = include_bonds

    def _emit_cancelled_if_requested(self) -> bool:
        if self.isInterruptionRequested():
            self.cancelled.emit()
            return True
        return False

    @staticmethod
    def _extract_rdkit_mol(mol_obj):
        if isinstance(mol_obj, ChemMol):
            return getattr(mol_obj, "mol", None)
        if RDKIT_AVAILABLE and isinstance(mol_obj, Chem.Mol):
            return mol_obj
        return None

    def run(self):
        try:
            if not self.mols:
                self.result_ready.emit(None)
                return

            all_counts = []
            global_support = Counter()
            total = len(self.mols)

            for i, mol_obj in enumerate(self.mols):
                if self._emit_cancelled_if_requested():
                    return
                if total > 50 and i % (total // 50) == 0:
                    self.progress.emit(int(i / total * 60))

                mol = self._extract_rdkit_mol(mol_obj)
                if not mol:
                    all_counts.append({})
                    continue

                # --- Descriptor Generation Logic ---
                mol_counts = Counter()

                # I - SEQUENCES (Paths)
                if self.topology == "I":
                    # Length in ISIDA usually refers to number of atoms. 
                    # RDKit paths are length of bonds. Atoms = Bonds + 1
                    for atom_len in range(self.min_len, self.max_len + 1):
                        bond_len = atom_len - 1
                        if bond_len < 0: continue

                        if bond_len == 0:
                            # Single atoms
                            if self.include_atoms:
                                for atom in mol.GetAtoms():
                                    mol_counts[atom.GetSymbol()] += 1
                            continue

                        # Paths
                        paths = Chem.FindAllPathsOfLengthN(mol, bond_len, useBonds=True)
                        for bond_indices in paths:
                            try:
                                # POPRAVEK: Določimo atome iz vezi, sicer RDKit vrne prazen niz
                                atom_indices = set()
                                for b_idx in bond_indices:
                                    bond = mol.GetBondWithIdx(b_idx)
                                    atom_indices.add(bond.GetBeginAtomIdx())
                                    atom_indices.add(bond.GetEndAtomIdx())

                                # Create a sub-molecule/SMILES for the path
                                frag_smi = Chem.MolFragmentToSmiles(
                                    mol, 
                                    bondsToUse=bond_indices, 
                                    atomsToUse=list(atom_indices), # <--- KLJUČNI POPRAVEK
                                    canonical=True,
                                    allBondsExplicit=self.include_bonds,
                                    allHsExplicit=False 
                                )
                                
                                if frag_smi:
                                    mol_counts[frag_smi] += 1
                            except Exception:
                                pass

                # II - ATOM CENTERED (Augmented Atoms / Shells)
                elif self.topology == "II":
                    # Radius 0 to N
                    for radius in range(self.min_len, self.max_len + 1):
                        for atom in mol.GetAtoms():
                            try:
                                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx())
                                # env so indeksi vezi.
                                
                                # Če je radius > 0 in ni okolice, preskočimo
                                if radius > 0 and not env: 
                                    continue
                                
                                # Če je radius 0, je env prazen, ampak to je ok (samo atom)
                                if radius == 0:
                                    smi = atom.GetSymbol()
                                else:
                                    # Ustvari submol iz vezi
                                    amap = {}
                                    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                                    smi = Chem.MolToSmiles(submol, canonical=True, allBondsExplicit=self.include_bonds)
                                
                                if smi:
                                    mol_counts[smi] += 1
                            except Exception:
                                pass

                # III - TRIPLETS (Distance based)
                elif self.topology == "III":
                    # Computationally expensive O(N^3)
                    dm = Chem.GetDistanceMatrix(mol)
                    atoms = list(mol.GetAtoms())
                    n_atoms = len(atoms)
                    
                    if n_atoms > 60: 
                        # Skip large molecules to avoid freezing UI for now
                        pass 
                    else:
                        # Combinations of 3
                        for idxs in itertools.combinations(range(n_atoms), 3):
                            if self._emit_cancelled_if_requested():
                                return
                            i, j, k = idxs
                            
                            d_ij = int(dm[i, j])
                            d_jk = int(dm[j, k])
                            d_ki = int(dm[k, i])
                            
                            # Filter by min/max
                            if (d_ij > self.max_len or d_jk > self.max_len or d_ki > self.max_len):
                                continue
                            if (d_ij < self.min_len and d_jk < self.min_len and d_ki < self.min_len):
                                continue

                            # Standard ISIDA triplet format usually: A(d1)B(d2)C(d3)
                            # Simplified descriptor: Sorted atom symbols + Sorted distances
                            # To ensures canonical representation (A-B-C is same as C-B-A)
                            syms = sorted([atoms[i].GetSymbol(), atoms[j].GetSymbol(), atoms[k].GetSymbol()])
                            dists = sorted([d_ij, d_jk, d_ki])
                            
                            desc = f"{syms[0]}.{syms[1]}.{syms[2]}-{dists[0]}.{dists[1]}.{dists[2]}"
                            mol_counts[desc] += 1

                all_counts.append(mol_counts)
                for frag in mol_counts:
                    global_support[frag] += 1

                # --- Filter & Matrix Construction ---
            if self._emit_cancelled_if_requested():
                return
            self.progress.emit(70)
            vocabulary = sorted(
                [f for f, count in global_support.items() if count >= self.min_support]
            )

            n_samples = len(all_counts)
            n_features = len(vocabulary)
            
            # Using float32 for counts to save memory if matrix is large
            X = np.zeros((n_samples, n_features), dtype=np.float32)
            vocab_index = {frag: i for i, frag in enumerate(vocabulary)}
            
            for i, mc in enumerate(all_counts):
                if self._emit_cancelled_if_requested():
                    return
                if total > 100 and i % (total // 100) == 0:
                    self.progress.emit(70 + int(i / total * 30))
                for frag, count in mc.items():
                    if frag in vocab_index:
                        X[i, vocab_index[frag]] = count
            
            self.progress.emit(100)
            self.result_ready.emit((vocabulary, X))

        except Exception as e:
            logger.exception("ISIDA descriptor generation failed.")
            self.error_occurred.emit(str(e))


class OWIsidaDescriptors(widget.OWWidget, ConcurrentWidgetMixin):
    name = "ISIDA Descriptors"
    description = "Generate fragment counts (ISIDA-like sequences, shells, triplets)."
    icon = "icons/descriptors/owisdidadescriptorwidget.png"
    priority = 133
    keywords = ["isida", "fragment", "descriptors", "smiles"]

    class Inputs:
        data = widget.Input("Data", Table)
        molecules = widget.Input("Molecules", list, auto_summary=False)

    class Outputs:
        data = widget.Output("Data", Table)

    # Settings
    topology_index: int = Setting(0) # 0=Sequences, 1=AtomCentered, 2=Triplets
    min_length: int = Setting(2)
    max_length: int = Setting(4)
    min_support: int = Setting(2)
    include_atoms: bool = Setting(True)
    include_bonds: bool = Setting(True)
    append_features: bool = Setting(True)

    def __init__(self) -> None:
        widget.OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self._mols = []
        self._input_data = None
        self._worker: Optional[IsidaWorker] = None

        # --- GUI ---
        box = gui.widgetBox(self.controlArea, "Topology")
        self.combo_topology = QComboBox()
        self.combo_topology.addItems([
            "I - Sequences (Paths)", 
            "II - Atom Centered (Shells)", 
            "III - Triplets (Distances)"
        ])
        self.combo_topology.setCurrentIndex(self.topology_index)
        self.combo_topology.currentIndexChanged.connect(self._on_topology_changed)
        box.layout().addWidget(self.combo_topology)

        # Length / Radius Settings
        self.box_len = gui.widgetBox(self.controlArea, "Settings")
        
        # Manual layout for SpinBox with Labels to allow text updates
        h1 = QHBoxLayout()
        self.lbl_min = QLabel("Min Length/Radius:")
        self.spin_min = gui.spin(self.box_len, self, "min_length", 0, 15, controlWidth=60)
        self.spin_min.setParent(None) # Remove from auto-layout
        h1.addWidget(self.lbl_min)
        h1.addWidget(self.spin_min)
        self.box_len.layout().addLayout(h1)

        h2 = QHBoxLayout()
        self.lbl_max = QLabel("Max Length/Radius:")
        self.spin_max = gui.spin(self.box_len, self, "max_length", 1, 20, controlWidth=60)
        self.spin_max.setParent(None)
        h2.addWidget(self.lbl_max)
        h2.addWidget(self.spin_max)
        self.box_len.layout().addLayout(h2)
        
        # Content
        box_cont = gui.widgetBox(self.controlArea, "Content")
        gui.checkBox(box_cont, self, "include_atoms", "Include Atoms (A)", callback=self._check_runnable)
        gui.checkBox(box_cont, self, "include_bonds", "Include Bonds (B)", callback=self._check_runnable)

        gui.separator(self.controlArea)
        
        # Filtering
        box_filter = gui.widgetBox(self.controlArea, "Filtering")
        gui.spin(box_filter, self, "min_support", 1, 1000, label="Min support (mols):",
                 tooltip="Fragment must appear in at least X molecules to be included.")
        gui.checkBox(box_filter, self, "append_features", "Append to input data")

        self.btn_compute = gui.button(self.controlArea, self, "Compute Descriptors", callback=self._start_computation)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)

        self.info_label = QLabel("No data.")
        self.controlArea.layout().addWidget(self.info_label)

        if not RDKIT_AVAILABLE:
            set_widget_error(self, "RDKit is not installed.")
            self.btn_compute.setEnabled(False)

        self._on_topology_changed()

    def _on_topology_changed(self):
        self.topology_index = self.combo_topology.currentIndex()
        if self.topology_index == 0: # Sequences
            self.lbl_min.setText("Min Atom Length:")
            self.lbl_max.setText("Max Atom Length:")
        elif self.topology_index == 1: # Atom Centered
            self.lbl_min.setText("Min Radius:")
            self.lbl_max.setText("Max Radius:")
        else: # Triplets
            self.lbl_min.setText("Min Distance:")
            self.lbl_max.setText("Max Distance:")

    def _check_runnable(self):
        pass

    # ---------------- Inputs ----------------
    
    @Inputs.data
    def set_data(self, data: Optional[Table]) -> None:
        self._input_data = data
        if data:
            self._mols = self._extract_mols_from_table(data)
            self.info_label.setText(f"Input: {len(self._mols)} molecules (Table).")
        else:
            if not self._mols: 
                self.info_label.setText("No data.")
        self.Outputs.data.send(None)

    @Inputs.molecules
    def set_molecules(self, mols: Optional[List[ChemMol]]) -> None:
        if mols:
            self._mols = mols
            self._input_data = None 
            self.info_label.setText(f"Input: {len(self._mols)} molecules (List).")
        else:
            if self._input_data:
                self._mols = self._extract_mols_from_table(self._input_data)
                self.info_label.setText(f"Input: {len(self._mols)} molecules (Table).")
            else:
                self._mols = []
                self.info_label.setText("No data.")
        self.Outputs.data.send(None)

    # ---------------- Logic ----------------

    def _extract_mols_from_table(self, data: Table) -> List[Chem.Mol]:
        mols = []
        if not data: return []
        
        smiles_var = None
        candidates = ["smiles", "canonical_smiles", "structure", "can", "smi"]
        
        for var in data.domain.metas:
            if var.name.strip().lower() in candidates:
                smiles_var = var
                break
        
        if not smiles_var:
            for var in data.domain.metas:
                if isinstance(var, StringVariable):
                    smiles_var = var
                    break
        
        if not smiles_var:
            set_widget_warning(self, "Could not find SMILES column.")
            return []

        for row in data:
            try:
                val = str(row[smiles_var])
                if val and val != "?" and val.lower() != "nan":
                    m = safe_mol_from_smiles(val, sanitize=True, remove_hs=True).mol
                    mols.append(m) 
                else:
                    mols.append(None)
            except Exception:
                mols.append(None)
        return mols

    def _start_computation(self):
        if not self._mols:
            set_widget_warning(self, "No molecules.")
            return
        if self._worker is not None and self._worker.isRunning():
            set_widget_warning(self, "Descriptor computation is already running.")
            return
        
        clear_widget_messages(self)
        self.btn_compute.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.info_label.setText("Computing descriptors…")
        
        topo_map = {0: "I", 1: "II", 2: "III"}
        topo = topo_map.get(self.topology_index, "I")

        self._worker = IsidaWorker(
            self._mols, 
            topo,
            self.min_length, 
            self.max_length, 
            self.min_support,
            self.include_atoms,
            self.include_bonds
        )
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.result_ready.connect(self._on_worker_finished)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _on_worker_finished(self, result):
        self._worker = None
        self.btn_compute.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if result is None:
            self.info_label.setText("No descriptor output was generated.")
            return

        vocabulary, X = result
        if len(vocabulary) == 0:
            set_widget_warning(self, f"No fragments found. (Min Support = {self.min_support}). Try reducing support.")
            self.Outputs.data.send(None)
            self.info_label.setText("No descriptors matched the current support threshold.")
            return

        # Create Table
        topo_label = ["Seq", "Shell", "Tri"][self.topology_index]
        new_attrs = [ContinuousVariable(name=f"{topo_label}_{frag}") for frag in vocabulary]
        
        if self._input_data and self.append_features:
            old_domain = self._input_data.domain
            new_domain = Domain(
                old_domain.attributes + tuple(new_attrs),
                old_domain.class_vars,
                old_domain.metas
            )
            X_old = self._input_data.X
            
            if X_old.shape[0] != X.shape[0]:
                set_widget_error(self, "Data length mismatch. Please reload data.")
                return

            X_new = np.hstack((X_old, X))
            new_table = Table(new_domain, X_new, self._input_data.Y, self._input_data.metas)
        else:
            domain = Domain(new_attrs)
            new_table = Table.from_numpy(domain, X)

        self.info_label.setText(f"Generated {len(vocabulary)} descriptors.")
        self.Outputs.data.send(new_table)

    def _on_worker_error(self, msg):
        self._worker = None
        self.btn_compute.setEnabled(True)
        self.progress_bar.setVisible(False)
        set_widget_error(self, msg)
        self.info_label.setText("Descriptor computation failed.")

    def _on_worker_cancelled(self):
        self._worker = None
        self.btn_compute.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.info_label.setText("Descriptor computation cancelled.")

    def onDeleteWidget(self):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
        super().onDeleteWidget()


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWIsidaDescriptors).run()
