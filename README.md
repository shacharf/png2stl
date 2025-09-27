# png2stl
Convert png as height map to stl

```bash
 png2stl.py --image  im.png --size 40 40 --height 4 --outstl model.stl
```

# Changlog
* 22-02-21 - Added support for non-rectangular shapes
  	     Have a "bug" of the size of the shape

# Development

Set up a virtual environment and install runtime dependencies:

```bash
python -m venv .venv-png2stl
source .venv-png2stl/bin/activate
pip install -r requirements.txt
```

Install and run the pre-commit hooks before committing changes:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The hook set runs `black`, `ruff`, and `mypy` so the codebase stays formatted, linted, and type-checked automatically.

# TODO
* Fix bug - does not seem to generate watertight geometry
* flip normals
* Documentation
* Add tests
