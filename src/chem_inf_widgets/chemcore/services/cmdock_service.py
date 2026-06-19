"""CmDock docking service.

A thin, GUI-free wrapper around the ``cmdock`` executable. The core
(:func:`dock_dataframe`) takes a :class:`pandas.DataFrame` of ligands and returns
one row per docked pose, so it can be unit-tested without Orange or a real
receptor. The Orange widget (:mod:`chem_inf_widgets.widgets.ow_cmdock_docking`)
converts the input/output tables around it.

Ported from the KNIME CmDock extension (``cmdock_runner.py``). CmDock itself is
installed separately; see https://gitlab.com/Jukic/cmdock.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

# rDock/CmDock score fields commonly present on each pose.
SCORE_FIELDS = ("SCORE", "SCORE.INTER", "SCORE.INTRA", "SCORE.RESTR")

# Fixed, non-property columns emitted for every pose row.
CORE_COLUMNS = (
    "input_row_id",
    "input_index",
    "ligand_name",
    "pose_index",
    "rank_by_score",
    "status",
    "score",
    "score_field",
    "score_total",
    "score_inter",
    "score_intra",
    "score_restr",
    "pose_sdf",
    "properties_json",
    "result_sdf_path",
    "cmdock_command",
    "cmdock_return_code",
    "cmdock_stdout",
    "cmdock_stderr",
    "error_message",
)


@dataclass(frozen=True)
class DockingSettings:
    """Everything needed to dock a table of ligands with CmDock."""

    cmdock_executable: str
    receptor_prm: str
    library_directory: str = ""
    protocol_file: str = "dock.prm"
    n_docking_runs: int = 100
    n_best_poses: int = 5
    score_tag: str = "SCORE.INTER"
    input_mode: str = "Auto"  # "Auto" | "SDF text" | "SDF file path"
    extra_flags: str = ""
    working_directory: str = ""
    keep_temporary_files: bool = False
    fail_on_ligand_error: bool = False
    use_gnu_parallel: bool = False
    gnu_parallel_executable: str = "parallel"
    parallel_jobs: int = 2


@dataclass(frozen=True)
class SdfRecord:
    """One molecule/pose record parsed from an SDF file."""

    name: str
    sdf_text: str
    properties: dict[str, str]


@dataclass(frozen=True)
class DockingJob:
    """One prepared CmDock job."""

    input_row_id: str
    input_index: int
    passthrough_values: dict[str, Any]
    ligand_file: Path
    output_file: Path
    command: list[str]
    stdout_file: Path
    stderr_file: Path
    return_code_file: Path


def output_columns(passthrough_columns: Iterable[str] | None = None) -> list[str]:
    """The fixed output columns, optionally followed by pass-through columns."""

    columns = list(CORE_COLUMNS)
    if passthrough_columns:
        columns.extend(passthrough_columns)
    return columns


# --------------------------------------------------------------------------- #
# SDF parsing (no chemistry dependency)
# --------------------------------------------------------------------------- #
def parse_sdf_records(sdf_text: str) -> list[SdfRecord]:
    """Parse SDF records and their ``> <key>`` property blocks.

    Preserves each complete record including the terminating ``$$$$`` delimiter.
    """

    records: list[SdfRecord] = []
    for chunk in sdf_text.split("$$$$"):
        chunk = chunk.strip("\n\r")
        if not chunk.strip():
            continue

        lines = chunk.splitlines()
        name = lines[0].strip() if lines else ""
        props: dict[str, str] = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith(">") and "<" in line and ">" in line:
                start = line.find("<") + 1
                end = line.find(">", start)
                key = line[start:end].strip()
                i += 1
                value_lines: list[str] = []
                while i < len(lines) and lines[i].strip() != "":
                    value_lines.append(lines[i].rstrip())
                    i += 1
                props[key] = "\n".join(value_lines)
            i += 1

        records.append(SdfRecord(name=name, sdf_text=chunk + "\n$$$$\n", properties=props))

    return records


def get_property_case_insensitive(properties: dict[str, str], key: str) -> str | None:
    """Return an SDF property value independent of key casing."""

    wanted = key.lower()
    for prop_key, value in properties.items():
        if prop_key.lower() == wanted:
            return value
    return None


def parse_float(value: Any) -> float | None:
    """Convert an SDF numeric property to float, or None if not numeric."""

    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Executable / receptor / environment resolution
# --------------------------------------------------------------------------- #
def resolve_executable(executable: str) -> str:
    """Resolve a path or PATH entry to the ``cmdock`` executable."""

    executable = str(executable).strip()
    if not executable:
        raise ValueError("CmDock executable is not configured.")

    expanded = Path(os.path.expanduser(executable))
    if expanded.exists():
        return str(expanded)
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    raise FileNotFoundError(f"CmDock executable not found: {executable}")


def resolve_parallel_executable(executable: str) -> str:
    """Resolve the GNU Parallel executable path."""

    executable = str(executable or "parallel").strip() or "parallel"
    expanded = Path(os.path.expanduser(executable))
    if expanded.exists():
        return str(expanded)
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    raise FileNotFoundError(f"GNU Parallel executable not found: {executable}")


def receptor_prm_path(path: str) -> Path:
    """Validate and normalize the receptor ``.prm`` file path."""

    prm = Path(os.path.expanduser(str(path).strip())).resolve()
    if not prm.is_file():
        raise FileNotFoundError(f"Receptor .prm file not found: {prm}")
    return prm


def referenced_receptor_file(prm: Path) -> "str | None":
    """Return the ``RECEPTOR_FILE`` value declared in a CmDock ``.prm`` (or None)."""

    try:
        for line in prm.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("RECEPTOR_FILE"):
                value = stripped[len("RECEPTOR_FILE"):].split("#")[0].strip()
                return value or None
    except OSError:
        return None
    return None


def library_path_variable() -> str:
    """Return the platform-specific dynamic library search-path variable."""

    system = platform.system().lower()
    if system == "windows":
        return "PATH"
    if system == "darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def infer_library_directories(executable: str) -> list[Path]:
    """Infer likely CmDock library directories from the executable path."""

    exe_path = Path(executable)
    if not exe_path.is_absolute():
        resolved = shutil.which(executable)
        if not resolved:
            return []
        exe_path = Path(resolved)

    candidates = [
        exe_path.parent.parent / "lib",
        exe_path.parent.parent / "lib64",
        exe_path.parent / "lib",
    ]
    return [path.resolve() for path in candidates if path.is_dir()]


def build_subprocess_environment(settings: DockingSettings, executable: str) -> dict[str, str]:
    """Build the CmDock subprocess environment with library and root paths."""

    env = os.environ.copy()
    var_name = library_path_variable()

    lib_dirs: list[Path] = []
    if settings.library_directory.strip():
        configured = Path(os.path.expanduser(settings.library_directory.strip())).resolve()
        if not configured.is_dir():
            raise FileNotFoundError(f"CmDock library directory not found: {configured}")
        lib_dirs.append(configured)
    lib_dirs.extend(infer_library_directories(executable))

    if lib_dirs:
        existing = env.get(var_name, "")
        values = [str(path) for path in lib_dirs]
        if existing:
            values.append(existing)
        env[var_name] = os.pathsep.join(values)

    cm_root = Path(executable).resolve().parent.parent
    if cm_root.is_dir():
        env.setdefault("CMDOCK_ROOT", str(cm_root))

    return env


# --------------------------------------------------------------------------- #
# Ligand input + command building
# --------------------------------------------------------------------------- #
def _looks_like_path(text: str) -> bool:
    """A filesystem path is a single, reasonably short line — SDF text is not."""

    return bool(text) and "\n" not in text and len(text) <= 4096


def _safe_is_file(path: Path) -> bool:
    """``Path.is_file()`` that returns False instead of raising on odd inputs.

    A long/multi-line SDF string used as a path triggers ``OSError`` (e.g.
    ENAMETOOLONG) rather than a clean ``False``.
    """

    try:
        return path.is_file()
    except OSError:
        return False


def read_ligand_input(value: Any, input_mode: str) -> str:
    """Read SDF text from a table cell.

    ``input_mode`` is ``Auto``, ``SDF text`` or ``SDF file path``. ``Auto``
    treats an existing path as a file and everything else as SDF text.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("Missing ligand SDF value.")

    text = str(value)
    mode = input_mode.strip().lower()
    stripped = text.strip()
    path = Path(os.path.expanduser(stripped)) if _looks_like_path(stripped) else None

    use_file = mode == "sdf file path" or (mode == "auto" and path is not None and _safe_is_file(path))
    if use_file:
        if path is None or not _safe_is_file(path):
            raise FileNotFoundError(f"Ligand SDF file not found: {stripped}")
        return path.read_text(encoding="utf-8", errors="replace")

    if "$$$$" not in text:
        raise ValueError("Ligand input is not SDF text and is not an existing file path.")
    return text


def write_ligand_file(work_dir: Path, input_index: int, sdf_text: str) -> Path:
    """Write one input ligand SDF to a temporary file for CmDock."""

    ligand_file = work_dir / f"ligand_{input_index}.sdf"
    ligand_file.write_text(sdf_text, encoding="utf-8")
    return ligand_file


def build_cmdock_command(
    settings: DockingSettings,
    executable: str,
    receptor_prm: Path,
    ligand_file: Path,
    output_file: Path,
) -> list[str]:
    """Build the CmDock argv (no shell)."""

    command = [
        executable,
        "-r", receptor_prm.name,
        "-p", settings.protocol_file.strip() or "dock.prm",
        "-i", str(ligand_file),
        "-n", str(int(settings.n_docking_runs)),
    ]
    if int(settings.n_best_poses) > 0:
        command.extend(["-b", str(int(settings.n_best_poses))])
    command.extend(["-o", str(output_file)])
    if settings.extra_flags.strip():
        command.extend(shlex.split(settings.extra_flags))
    return command


# --------------------------------------------------------------------------- #
# Row building
# --------------------------------------------------------------------------- #
def make_pose_rows(
    job: DockingJob,
    records: Iterable[SdfRecord],
    result_sdf_path: Path | None,
    return_code: int,
    stdout: str,
    stderr: str,
    score_tag: str,
) -> list[dict[str, Any]]:
    """Convert parsed SDF poses into output rows, ranked by score (ascending)."""

    rows: list[dict[str, Any]] = []
    score_key = score_tag.strip() or "SCORE.INTER"
    for pose_index, record in enumerate(records, start=1):
        score = parse_float(get_property_case_insensitive(record.properties, score_key))
        row = {
            "input_row_id": job.input_row_id,
            "input_index": job.input_index,
            "ligand_name": record.name,
            "pose_index": pose_index,
            "rank_by_score": 0,
            "status": "ok",
            "score": score,
            "score_field": score_key,
            "score_total": parse_float(get_property_case_insensitive(record.properties, "SCORE")),
            "score_inter": parse_float(get_property_case_insensitive(record.properties, "SCORE.INTER")),
            "score_intra": parse_float(get_property_case_insensitive(record.properties, "SCORE.INTRA")),
            "score_restr": parse_float(get_property_case_insensitive(record.properties, "SCORE.RESTR")),
            "pose_sdf": record.sdf_text,
            "properties_json": json.dumps(record.properties, sort_keys=True),
            "result_sdf_path": str(result_sdf_path) if result_sdf_path else "",
            "cmdock_command": shlex.join(job.command),
            "cmdock_return_code": return_code,
            "cmdock_stdout": stdout,
            "cmdock_stderr": stderr,
            "error_message": "",
            "_properties": dict(record.properties),  # expanded to columns by the widget
        }
        row.update(job.passthrough_values)
        rows.append(row)

    rows.sort(key=lambda r: (r["score"] is None, r["score"] if r["score"] is not None else 0.0))
    for rank, row in enumerate(rows, start=1):
        row["rank_by_score"] = rank
    return rows


def make_error_row(
    input_row_id: str,
    input_index: int,
    passthrough_values: dict[str, Any],
    message: str,
    command: list[str] | None = None,
    return_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    result_sdf_path: Path | None = None,
    score_tag: str = "SCORE.INTER",
) -> dict[str, Any]:
    """Create an output row representing a failed ligand."""

    row = {
        "input_row_id": input_row_id,
        "input_index": input_index,
        "ligand_name": "",
        "pose_index": 0,
        "rank_by_score": 0,
        "status": "failed",
        "score": None,
        "score_field": score_tag,
        "score_total": None,
        "score_inter": None,
        "score_intra": None,
        "score_restr": None,
        "pose_sdf": "",
        "properties_json": "{}",
        "result_sdf_path": str(result_sdf_path) if result_sdf_path else "",
        "cmdock_command": shlex.join(command or []),
        "cmdock_return_code": return_code,
        "cmdock_stdout": stdout,
        "cmdock_stderr": stderr,
        "error_message": message,
        "_properties": {},
    }
    row.update(passthrough_values)
    return row


def result_rows_for_job(
    job: DockingJob, settings: DockingSettings, return_code: int, stdout: str, stderr: str
) -> list[dict[str, Any]]:
    """Convert one finished job into success or error rows."""

    persistent = job.output_file if settings.keep_temporary_files else None

    def error(message: str) -> list[dict[str, Any]]:
        return [make_error_row(
            job.input_row_id, job.input_index, job.passthrough_values, message,
            job.command, return_code, stdout, stderr, persistent, settings.score_tag,
        )]

    if return_code != 0:
        return error(f"CmDock failed with return code {return_code}.")
    if not job.output_file.is_file():
        return error("CmDock finished but did not create an output SDF file.")

    records = parse_sdf_records(job.output_file.read_text(encoding="utf-8", errors="replace"))
    if not records:
        return error("CmDock output SDF contained no pose records.")

    return make_pose_rows(job, records, persistent, return_code, stdout, stderr, settings.score_tag)


# --------------------------------------------------------------------------- #
# GNU Parallel support
# --------------------------------------------------------------------------- #
def shell_line_for_parallel_job(job: DockingJob) -> str:
    """One shell command line for GNU Parallel (captures stdout/stderr/rc)."""

    command = shlex.join(job.command)
    stdout = shlex.quote(str(job.stdout_file))
    stderr = shlex.quote(str(job.stderr_file))
    return_code = shlex.quote(str(job.return_code_file))
    return f"{command} > {stdout} 2> {stderr}; printf '%s\\n' $? > {return_code}"


def run_jobs_with_gnu_parallel(
    jobs: list[DockingJob],
    settings: DockingSettings,
    prm_dir: Path,
    subprocess_env: dict[str, str],
    progress: Callable[[float, str], None] | None = None,
) -> None:
    """Run prepared jobs through GNU Parallel."""

    if not jobs:
        return

    parallel_executable = resolve_parallel_executable(settings.gnu_parallel_executable)
    parallel_jobs = max(1, int(settings.parallel_jobs))
    command = [parallel_executable, "--jobs", str(parallel_jobs), "--will-cite"]
    parallel_input = "\n".join(shell_line_for_parallel_job(job) for job in jobs) + "\n"

    if progress:
        progress(0.35, f"Running {len(jobs)} CmDock jobs with GNU Parallel ({parallel_jobs} workers)")

    result = subprocess.run(
        command, input=parallel_input, cwd=str(prm_dir),
        capture_output=True, text=True, check=False, env=subprocess_env,
    )
    if result.returncode != 0:
        missing = [job for job in jobs if not job.return_code_file.is_file()]
        if missing:
            raise RuntimeError(
                f"GNU Parallel failed with return code {result.returncode}. "
                f"{len(missing)} jobs did not produce return code files.\n{result.stderr}"
            )
    if progress:
        progress(0.75, "Collecting CmDock results")


def read_job_return_code(job: DockingJob) -> int:
    """Read the per-job return code emitted by a GNU Parallel shell line."""

    if not job.return_code_file.is_file():
        return 127
    try:
        return int(job.return_code_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return 127


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare_jobs(
    frame: pd.DataFrame,
    sdf_column: str,
    settings: DockingSettings,
    executable: str,
    prm: Path,
    work_dir: Path,
    passthrough_column_map: dict[str, str],
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[DockingJob], list[dict[str, Any]]]:
    """Prepare ligand files and commands before execution."""

    jobs: list[DockingJob] = []
    rows: list[dict[str, Any]] = []
    total = max(len(frame), 1)

    for offset, (row_id, row) in enumerate(frame.iterrows(), start=1):
        input_row_id = str(row_id)
        input_index = offset - 1
        output_file = work_dir / f"docked_{input_index}.sdf"
        passthrough_values = {
            output_name: row[input_name]
            for input_name, output_name in passthrough_column_map.items()
        }

        try:
            if progress:
                progress((offset - 1) / total * 0.25, f"Preparing ligand {offset} of {len(frame)}")
            ligand_sdf = read_ligand_input(row[sdf_column], settings.input_mode)
            ligand_file = write_ligand_file(work_dir, input_index, ligand_sdf)
            command = build_cmdock_command(settings, executable, prm, ligand_file, output_file)
            jobs.append(DockingJob(
                input_row_id=input_row_id,
                input_index=input_index,
                passthrough_values=passthrough_values,
                ligand_file=ligand_file,
                output_file=output_file,
                command=command,
                stdout_file=work_dir / f"docked_{input_index}.stdout.txt",
                stderr_file=work_dir / f"docked_{input_index}.stderr.txt",
                return_code_file=work_dir / f"docked_{input_index}.returncode.txt",
            ))
        except Exception as exc:
            if settings.fail_on_ligand_error:
                raise
            rows.append(make_error_row(
                input_row_id, input_index, passthrough_values, str(exc), None, None,
                result_sdf_path=output_file if settings.keep_temporary_files else None,
                score_tag=settings.score_tag,
            ))

    return jobs, rows


def _validate_settings(settings: DockingSettings, frame: pd.DataFrame, sdf_column: str,
                       passthrough_column_map: dict[str, str]) -> None:
    if sdf_column not in frame.columns:
        raise ValueError(f"Input SDF column not found: {sdf_column}")
    missing = [c for c in passthrough_column_map if c not in frame.columns]
    if missing:
        raise ValueError(f"Input pass-through column(s) not found: {', '.join(missing)}")
    if int(settings.n_docking_runs) < 1:
        raise ValueError("Number of docking runs must be at least 1.")
    if int(settings.n_best_poses) < 0:
        raise ValueError("Number of best poses must be zero or greater.")
    if int(settings.parallel_jobs) < 1:
        raise ValueError("Number of GNU Parallel jobs must be at least 1.")


def dock_dataframe(
    frame: pd.DataFrame,
    sdf_column: str,
    settings: DockingSettings,
    passthrough_column_map: dict[str, str] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Dock all ligands from a DataFrame and return a list of pose-row dicts.

    Each row dict carries the fixed :data:`CORE_COLUMNS` plus a ``_properties``
    dict of the raw SDF property block, which the widget expands into columns.
    """

    passthrough_column_map = passthrough_column_map or {}
    _validate_settings(settings, frame, sdf_column, passthrough_column_map)

    executable = resolve_executable(settings.cmdock_executable)
    prm = receptor_prm_path(settings.receptor_prm)
    prm_dir = prm.parent
    subprocess_env = build_subprocess_environment(settings, executable)

    parent_work_dir = (
        Path(os.path.expanduser(settings.working_directory)).resolve()
        if settings.working_directory.strip()
        else Path(tempfile.gettempdir())
    )
    parent_work_dir.mkdir(parents=True, exist_ok=True)

    temp_context: tempfile.TemporaryDirectory[str] | None = None
    if settings.keep_temporary_files:
        work_dir = Path(tempfile.mkdtemp(prefix="owcheminf_cmdock_", dir=str(parent_work_dir)))
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="owcheminf_cmdock_", dir=str(parent_work_dir))
        work_dir = Path(temp_context.name)

    rows: list[dict[str, Any]] = []
    try:
        jobs, rows = prepare_jobs(
            frame, sdf_column, settings, executable, prm, work_dir, passthrough_column_map, progress,
        )

        if settings.use_gnu_parallel and jobs:
            run_jobs_with_gnu_parallel(jobs, settings, prm_dir, subprocess_env, progress)
            for job in jobs:
                return_code = read_job_return_code(job)
                stdout = job.stdout_file.read_text(encoding="utf-8", errors="replace") if job.stdout_file.is_file() else ""
                stderr = job.stderr_file.read_text(encoding="utf-8", errors="replace") if job.stderr_file.is_file() else ""
                if settings.fail_on_ligand_error and return_code != 0:
                    raise RuntimeError(f"CmDock failed for row {job.input_row_id} (rc {return_code}).\n{stderr}")
                rows.extend(result_rows_for_job(job, settings, return_code, stdout, stderr))
        else:
            total = max(len(jobs), 1)
            for offset, job in enumerate(jobs, start=1):
                if progress:
                    progress(0.25 + ((offset - 1) / total * 0.65), f"Docking ligand {offset} of {len(jobs)}")
                result = subprocess.run(
                    job.command, cwd=str(prm_dir), capture_output=True, text=True,
                    check=False, env=subprocess_env,
                )
                if settings.fail_on_ligand_error and result.returncode != 0:
                    raise RuntimeError(f"CmDock failed for row {job.input_row_id} (rc {result.returncode}).\n{result.stderr}")
                rows.extend(result_rows_for_job(job, settings, result.returncode, result.stdout, result.stderr))

        if progress:
            progress(1.0, "CmDock docking finished")
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    return rows
