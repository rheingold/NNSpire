# LICENSES/third-party — Third-party license texts

This directory must contain a copy of every license and attribution notice for
third-party libraries that are distributed with NNStudio (embedded or shipped
alongside the binary). It is displayed in the Studio `Help → About → Licenses`
dialog.

**Status: placeholder — files added as each dependency is vendored/fetched.**

---

## Required files — add when the dependency is vendored or linked

| File to add | Dependency | Trigger |
|---|---|---|
| `Qt6-LGPL-v3.txt` | Qt 6 | Any binary release |
| `Python-PSF-LICENSE.txt` | Embeddable Python | Any binary release |
| `MinGW-w64-COPYING.txt` | MinGW-w64 runtime | Windows binary release |
| `pybind11-MIT.txt` | pybind11 | Build-time only — include in source dist |
| `onnx-LICENSE` | ONNX protobuf (Apache 2.0) | When ONNX I/O ships |
| `onnx-NOTICE` | ONNX protobuf | When ONNX I/O ships |
| `Eigen-MPL-2.0.txt` | Eigen 3.4 | When Eigen is vendored |
| `OpenSSL-LICENSE.txt` | OpenSSL 3.x | Any binary release that uses TLS |
| `GoogleTest-LICENSE` | GoogleTest (BSD-3-Clause) | Test builds only — do NOT ship in release |
| `Qiskit-LICENSE` | IBM Qiskit (Apache 2.0) | **When Phase 6 quantum backend ships** |
| `Qiskit-NOTICE` | IBM Qiskit | When Phase 6 quantum backend ships |
| `NVIDIA-CUDA-EULA.txt` | NVIDIA CUDA Toolkit | **When Phase 4 CUDA backend ships** — link to EULA or include notice; review NVIDIA redistribution terms carefully before shipping any CUDA library files |

---

## Notes

- The **CUDA Toolkit EULA** is proprietary and may not permit redistribution of
  cuBLAS/cuDNN libraries directly. The recommended approach is to declare a
  runtime dependency on the user's locally-installed CUDA installation and NOT
  vendor NVIDIA binaries. Verify this before any release that links CUDA.
- **Modified Eigen files** must remain under MPL 2.0 — if you modify any Eigen
  header, you must publish the modified file under MPL 2.0 (file-level copyleft).
- All Apache 2.0 dependencies (ONNX, OpenSSL, Qiskit) require inclusion of
  their `NOTICE` file in addition to the license text.
