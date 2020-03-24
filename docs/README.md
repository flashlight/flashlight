# Flashlight Documentation

## Building the Docs

To build the documentation, follow the steps below.

### Setup (do once)

Install [Doxygen](http://www.doxygen.nl/manual/install.html).

Install sphinx, breathe and the theme using the `requirements.txt` file in `docs/`:

```
pip install -r requirements.txt
```

### Build

From `docs/`:

```
doxygen && make html
```

If you run into issues rebuilding the docs, run `make clean` before building again.

### View the Docs

Run a server in `docs/build/html`:

```
python -m http.server <port>
```

Point browser to `http://localhost:<port>`.
