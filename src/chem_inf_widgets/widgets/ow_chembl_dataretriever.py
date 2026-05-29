import requests
import pandas as pd
import logging
from datetime import datetime

from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets import gui
from Orange.data import Table, Domain, StringVariable, ContinuousVariable

# Qt imports via AnyQt for Orange compatibility
from AnyQt.QtWidgets import QPlainTextEdit, QSizePolicy, QApplication
from AnyQt.QtCore import Qt, pyqtSlot, pyqtSignal
from Orange.widgets.utils.concurrent import ThreadExecutor, methodinvoke

from chem_inf_widgets.chemcore.services.chembl_bioactivity_dataframe_service import (
    calculate_drug_properties,
    fetch_bioactivity_dataframe,
    filter_output_columns,
    normalize_smiles_column,
    process_ic50_values,
)
from chem_inf_widgets.chemcore.services.from_orange import table_to_chemmols_with_report
from chem_inf_widgets.widgets.ui_helpers import (
    clear_widget_messages,
    format_error_status,
    format_table_report,
    set_widget_error,
    set_widget_warning,
)

class ChEMBLBioactivityWidget(OWWidget):
    """Orange widget to fetch ChEMBL bioactivity data with drug properties."""
    
    name = "ChEMBL Bioactivity Retriever"
    description = "Fetches bioactivity data with drug design properties"
    icon = "icons/data_retrieval/chembl.png"
    priority = 103

    # Declare a signal to send log messages from any thread.
    logMessage = pyqtSignal(str)

    class Outputs:
        output_data = Output("Bioactivity Data", Table)

    def __init__(self):
        super().__init__()
        # Hide the main area so everything is in the control panel
        self.mainArea.hide()

        self.target_id = ""
        # Set up a logger for detailed debug/info messages.
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Connect the logMessage signal to the log_status slot
        self.logMessage.connect(self.log_status)

        # Build the entire UI in the control area
        self._build_ui()
        # Set up an executor for background processing
        self.executor = ThreadExecutor(self)
        self._future = None

    def _build_ui(self):
        """Construct the user interface in the control area."""
        control_box = gui.widgetBox(self.controlArea, orientation=Qt.Vertical, spacing=6)

        input_box = gui.widgetBox(control_box, "Retrieve Bioactivity Data", orientation=Qt.Vertical)
        gui.label(input_box, self, "Enter ChEMBL Target ID (e.g., CHEMBL2095150):")
        gui.lineEdit(input_box, self, "target_id", placeholderText="CHEMBLxxxxxx")
        self.fetch_button = gui.button(input_box, self, "Fetch Data", callback=self.fetch_bioactivity_data)

        self.status_label = gui.label(input_box, self, "Status: Awaiting input.")
        control_box.layout().addStretch(1)

        log_box = gui.widgetBox(control_box, "Status Log", orientation=Qt.Vertical)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_box.layout().addWidget(self.log_text)

    @pyqtSlot(str)
    def log_status(self, message: str):
        """Update the status label and log widget with a timestamped message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        formatted_message = f"{timestamp} - INFO - {message}"
        self.status_label.setText(message)
        self.logger.info(message)
        current_text = self.log_text.toPlainText()
        new_text = f"{current_text}\n{formatted_message}" if current_text else formatted_message
        self.log_text.setPlainText(new_text)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        QApplication.processEvents()

    @pyqtSlot(bool)
    def set_fetch_button_enabled(self, enabled: bool):
        """Helper to enable or disable the fetch button (called from main thread)."""
        self.fetch_button.setEnabled(enabled)

    def fetch_bioactivity_data(self):
        """Start background data retrieval and processing."""
        clear_widget_messages(self)
        target_id = self.target_id.strip()
        if not target_id:
            set_widget_warning(self, "Please enter a ChEMBL Target ID.")
            self.logMessage.emit("Status: Please enter a ChEMBL Target ID.")
            return

        self.fetch_button.setEnabled(False)
        self.logMessage.emit(f"Status: Connecting to ChEMBL API for target {target_id}...")

        self._future = self.executor.submit(self._fetch_data_in_background, target_id)
        setattr(self._future, "_chembl_target_id", target_id)
        self._future.add_done_callback(self._on_fetch_complete)

    def _fetch_data_in_background(self, target_id: str) -> Table:
        """Background function that fetches and processes bioactivity data."""
        df = self._fetch_chembl_data(target_id)
        if df.empty:
            raise ValueError(f"No data found for {target_id}.")

        self.logMessage.emit("Status: Data fetched successfully. Processing IC50 values...")
        df = process_ic50_values(df)
        df = normalize_smiles_column(df)

        if 'SMILES' in df.columns:
            self.logMessage.emit("Status: Calculating drug properties...")
            df = calculate_drug_properties(df)

        df = filter_output_columns(df)
        table = self._create_orange_table(df)
        return table

    def _on_fetch_complete(self, future):
        """Callback when background processing completes; update output on main thread."""
        try:
            table = future.result()
        except Exception as e:
            methodinvoke(self, "_handle_error", (str,))(f"Error during fetching: {e}")
            table = None
        # Use methodinvoke to schedule updating the output on the main thread.
        methodinvoke(self, "_update_output_from_table", (Table,))(table)
        methodinvoke(self, "set_fetch_button_enabled", (bool,))(True)

    @pyqtSlot(Table)
    def _update_output_from_table(self, table: Table):
        """Slot to update output once fetching is complete."""
        if table is not None:
            try:
                _mols, report = table_to_chemmols_with_report(table)
                self.logMessage.emit(
                    f"Status: Retrieved {len(table)} records for {self.target_id}; "
                    f"{format_table_report(report, prefix='SMILES parse', valid_label='valid SMILES', include_smiles_column=False)}."
                )
            except Exception:
                self.logMessage.emit(f"Status: Retrieved {len(table)} records for {self.target_id}.")
        self.Outputs.output_data.send(table)

    def _fetch_chembl_data(self, target_id: str) -> pd.DataFrame:
        """Fetch bioactivity data from ChEMBL API."""
        try:
            self.logMessage.emit("Status: Sending request to ChEMBL API...")
            df = fetch_bioactivity_dataframe(target_id, standard_type="IC50", limit=1000, timeout=30)
            self.logMessage.emit("Status: Request successful. Processing response...")
            return df

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Network error during API connection: {str(e)}") from e

    def _create_orange_table(self, df: pd.DataFrame) -> Table:
        """Convert a pandas DataFrame to an Orange Table."""
        num_cols = [col for col in df.columns if col in [
            'pchembl_value', 'IC50_nM', 'hbd', 'hba', 'rotable_bonds',
            'mw', 'tpsa', 'logp', 'lipinski_deviations'
        ]]
        meta_cols = [col for col in df.columns if col not in num_cols]
        domain = Domain(
            [ContinuousVariable(col) for col in num_cols],
            metas=[StringVariable(col) for col in meta_cols]
        )
        X = df[num_cols].to_numpy(dtype=float)
        metas = df[meta_cols].to_numpy(dtype=object)
        return Table.from_numpy(domain, X=X, metas=metas)

    @pyqtSlot(str)
    def _handle_error(self, message: str):
        """Centralized error handling."""
        set_widget_error(self, message)
        self.logMessage.emit(format_error_status(message))
        self.Outputs.output_data.send(None)

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(ChEMBLBioactivityWidget).run()
