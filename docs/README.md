## Flashlight Documentation

To build the documentation follow the steps below.

### Setup (do once)

Install [Doxygen](http://www.doxygen.nl/manual/install.html).

Install sphinx, breathe and the theme:

```
pip install -r requirements.txt
```

### Build the Docs

From `docs/`

```
doxygen && make html
```

### View the Docs

Run a server in `docs/build/html`

```
python -m http.server <port>
```

Point browser to http://localhost:port
