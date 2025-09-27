# Repository Guidelines

## Project Structure & Module Organization
- `png2stl.py`: primary CLI for converting grayscale PNGs into STL meshes; orchestrates contour detection, heightfield construction, and STL export.
- `lightweight_3dprint.py`: exploratory notebook-style script for volumetric experiments; treat it as a sandbox and avoid coupling production changes to it.
- `requirements.txt`: minimal runtime dependencies (`opencv-python`, `numpy-stl`). Add development-only tooling under extras to keep installs lightweight.
- Expect STL outputs to live alongside the invoked PNG unless `--outstl` is provided. Store large sample assets under `assets/` (create if needed) and keep repository-friendly placeholders in `samples/`.

## Build, Test, and Development Commands
- `python -m venv .venv-png2stl && source .venv-png2stl/bin/activate`: standard virtual environment workflow before installing packages.
- `pip install -r requirements.txt`: installs runtime dependencies; pin new libraries before committing.
- `python png2stl.py --image samples/logo.png --size 40 40 --height 4 --pic_height 2 --outstl builds/logo.stl`: reference run covering the common flags; scale values when calibrating for printers.
- `python lightweight_3dprint.py`: launches the experimental marching-cubes demo; expect GUI output.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, snake_case for functions, and UpperCamelCase for classes (`Im2stl`).
- Keep argument names aligned with CLI options (`--pic_height` → `pic_height`).
- Prefer explicit numpy operations over implicit loops; annotate tricky math with concise comments.
- Gate debug logging behind an argument or `if __name__ == '__main__'` block to keep library imports clean.
- Use black coding style.
- Use python 3.12 or above and write pythonic code.
  - When possible, use native type hints as opposed to importing them from typing

## Testing Guidelines
- No automated suite exists yet; create `tests/` with `pytest` fixtures that load small grayscale fixtures and assert STL vertex counts, bounding boxes, and watertightness helpers.
- Use deterministic PNGs under `tests/data/`; document their provenance.
- Run `pytest -q` locally before opening a PR; include regression images if reproducing reported bugs.

## Commit & Pull Request Guidelines
- Match the concise imperative tone seen in history (`Support for non-rectangular shapes`, `Update README.md`). Keep subject ≤72 chars and expand motivation in the body when needed.
- Reference issue IDs in the subject or first body line when applicable.
- PRs should summarize functional changes, list validation commands, attach before/after renders when geometry changes, and flag any new dependencies.
