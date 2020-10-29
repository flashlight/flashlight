# Building flashlight Examples

Examples included with flashlight can be built either in-source (in a `build` directory) or out-of-source, after installing, as a standalone build.

In the `make install` step, examples are placed in `<prefix>/share/flashlight/examples`, where `<prefix>` is the `CMAKE_INSTALL_PREFIX` (on Linux, this is typically `/usr/local` by default).

## Building In-Source
Building in-source is simple; examples are built by default. Binaries are placed in `build/examples`.

To disable building examples in source, simply make sure `FL_BUILD_EXAMPLES` is `OFF`. Even though examples won't be built, the install step will still place source files in `examples/` in the install target directory mentioned above.

## Building as a Standalone Project
Examples can also be built as standalone projects outside of the `build` directory. After the installation step, simply copy the example source path to a suitable directory, and build:
```
mkdir -p ~/flashlight-examples
cp -r <prefix>/share/flashlight/examples ~/flashlight-examples
mkdir build && cd build
cmake ..
make -j4  # (or any number of threads)
```

If flashlight (or ArrayFire) were built and installed and installed with a custom `CMAKE_INSTALL_PREFIX`, then `-Dflashlight_DIR` (and `DArrayFire_DIR`) need to be set to the directory containing `flashlightConfig.cmake` (and `ArrayFireConfig.cmake`), so the targets can properly be imported. These flags must be passed to `cmake` when building.
