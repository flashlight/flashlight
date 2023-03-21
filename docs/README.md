# Flashlight Documentation

## Building the Docs

To build the documentation, follow the steps below.

### Setup

First, install [Doxygen](http://www.doxygen.nl/manual/install.html).

Install Sphinx, Breathe and the theme using the `requirements.txt` file in `docs`:
```bash
cd docs
pip install -r requirements.txt
```

### Building

From `docs`, run:
```bash
doxygen
make html
```

If you run into issues rebuilding docs, run `make clean` before building again.

### Viewing

After buildling, from the `docs` directory, run a local server to view artifacts:
```bash
python -m http.server <port> --directory build/html
```

Point your browser to `http://localhost:<port>` to view.
