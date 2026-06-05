# NNSpire — Third-party dependencies

This directory contains vendored third-party libraries.

Add each by running the commands below from the project root.
All header-only dependencies are safe to commit to git.
Binary dependencies go in `.gitignore`-listed locations.

## Add Eigen (header-only, MPL 2.0)
```bash
# Download and extract into third-party/eigen/
curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
     | tar xz --strip-components=1 -C NNSpire/third-party/eigen/
```

## Add pybind11 (header-only, MIT)
```bash
git submodule add https://github.com/pybind/pybind11.git NNSpire/third-party/pybind11
git submodule update --init
```

## Add GoogleTest (BSD 3-Clause, test builds only)
```bash
git submodule add https://github.com/google/googletest.git NNSpire/third-party/googletest
git submodule update --init
```

## Add ONNX protobuf (Apache 2.0)
```bash
# Via CMake FetchContent or vcpkg:
# vcpkg install onnx
# Or: git submodule add https://github.com/onnx/onnx.git NNSpire/third-party/onnx
```

## OpenSSL 3.x
Not vendored — use system OpenSSL on Linux/macOS; install via vcpkg on Windows:
```bash
vcpkg install openssl:x64-windows
```

## Status

| Library  | Version | License    | Status |
|----------|---------|------------|--------|
| Eigen    | 3.4.0   | MPL 2.0    | not yet fetched |
| pybind11 | 2.12+   | MIT        | not yet fetched |
| GoogleTest | latest | BSD 3-Cl  | not yet fetched |
| ONNX     | 1.16+   | Apache 2.0 | not yet fetched |
| OpenSSL  | 3.x     | Apache 2.0 | system / vcpkg  |
