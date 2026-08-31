"""Execute the committed notebooks and fail if any cell raises.

Notebooks rot silently. They are not imported by anything, so a rename or a
signature change in the modules they call breaks them without breaking a build,
and the failure is discovered whenever somebody next opens the file -- often
months later, often the person you were trying to impress.

Running them in CI makes that a build failure on the commit that caused it.

Marked as integration because executing a kernel is slower than a unit test.
Run the fast suite with: pytest -m "not integration"
"""

from __future__ import annotations

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_at_least_one_notebook_is_committed() -> None:
    """Guards against the directory silently emptying and the suite passing
    vacuously."""
    assert NOTEBOOKS, f"no notebooks found in {NOTEBOOK_DIR}"


@pytest.mark.integration
@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_executes_without_error(notebook_path: Path) -> None:
    """Execute every cell against the current source tree."""
    notebook = nbformat.read(notebook_path, as_version=4)

    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        # Run from the repository root so the notebook's relative data paths
        # resolve exactly as they do for a person running it there.
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()

    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, f"{notebook_path.name} raised: {errors[0].get('evalue')}"


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_defines_no_analysis_logic(notebook_path: Path) -> None:
    """Notebooks are a reading surface, not a place for logic.

    Anything worth defining is worth testing, and a function defined in a
    notebook is neither importable nor covered. This keeps the split honest
    rather than relying on discipline.
    """
    notebook = nbformat.read(notebook_path, as_version=4)

    offenders = [
        line.strip()
        for cell in notebook.cells
        if cell.cell_type == "code"
        for line in cell.source.splitlines()
        if line.startswith(("def ", "class "))
    ]

    assert not offenders, (
        f"{notebook_path.name} defines {offenders}; move it into src/zepto so it "
        "can be typed and tested"
    )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_imports_from_the_package(notebook_path: Path) -> None:
    """The point of the split: the notebook calls the same tested code that ships."""
    notebook = nbformat.read(notebook_path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "from zepto." in source
