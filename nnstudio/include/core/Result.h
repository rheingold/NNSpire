#pragma once
/* ============================================================================
 * Result.h — lightweight error-handling type (no exceptions in nnstudio-core)
 * LGPL v3
 *
 * WHY NO EXCEPTIONS?
 * ──────────────────
 * Exceptions are disabled in nnstudio-core (RTTI off, -fno-exceptions). Reason:
 * the engine is linked into plugin shared libraries loaded at runtime by an
 * untrusted host process. Exceptions crossing shared-library boundaries are
 * undefined behaviour in the C ABI (see ADR-011).
 *
 * THE PATTERN
 * ───────────
 *   Result<Tensor>  — either a Tensor value OR an Error (code + message string).
 *   Result<void>    — either success (no value) OR an Error (use VoidResult alias).
 *
 *   Check .ok() / .failed() before calling .value() or .error().
 *   .value() asserts in debug if called on a failed Result.
 *
 * NN_TRY(expr) — propagates an error upward (analogous to Rust's `?` operator):
 *   evaluates expr; if failed, returns its Error to the *caller* immediately.
 *   Only valid in functions that return Result<T> or Error.
 *
 * WHEN TO USE WHICH ErrorCode
 * ────────────────────────────
 *   ShapeMismatch   — tensor shapes incompatible for this operation
 *   DTypeMismatch   — operands have different dtypes
 *   DeviceMismatch  — tensors are on different devices (e.g. CPU vs CUDA)
 *   InvalidArgument — bad parameter value (negative size, null ptr, etc.)
 *   NotImplemented  — op in the interface but not yet in this backend
 *   BackendError    — backend-specific error (CUDA kernel failure, GPU OOM)
 *
 * ============================================================================ */

#include <variant>
#include <string>
#include <cassert>
#include <cstdint>
#include <utility>

namespace nnstudio::core {

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------
enum class ErrorCode : uint32_t {
    Ok = 0,
    InvalidArgument,
    OutOfMemory,
    DeviceMismatch,
    ShapeMismatch,
    DTypeMismatch,
    NotImplemented,
    IoError,
    PluginError,
    BackendError,
};

struct Error {
    ErrorCode   code    { ErrorCode::Ok };
    std::string message {};

    Error() = default;
    Error(ErrorCode c, std::string msg) : code(c), message(std::move(msg)) {}

    bool operator==(ErrorCode c) const noexcept { return code == c; }
    bool operator!=(ErrorCode c) const noexcept { return code != c; }
};

inline Error ok()                                      { return {}; }
inline Error err(ErrorCode c, std::string msg = {})    { return Error{c, std::move(msg)}; }

// ---------------------------------------------------------------------------
// Result<T>
// ---------------------------------------------------------------------------
template<typename T>
class Result {
public:
    // Construct from value (success)
    Result(T value)      : data_(std::in_place_index<0>, std::move(value)) {}   // NOLINT
    // Construct from error
    Result(Error error)  : data_(std::in_place_index<1>, std::move(error)) {}   // NOLINT

    bool ok()    const noexcept { return data_.index() == 0; }
    bool failed()const noexcept { return data_.index() == 1; }

    T&        value()       { assert(ok()); return std::get<0>(data_); }
    const T&  value() const { assert(ok()); return std::get<0>(data_); }

    Error&       error()       { assert(failed()); return std::get<1>(data_); }
    const Error& error() const { assert(failed()); return std::get<1>(data_); }

    T        valueOr(T def) const { return ok() ? std::get<0>(data_) : def; }

    // Unwrap — aborts on failure in debug; in release callers must check ok()
    T& operator*()        { return value(); }
    const T& operator*() const { return value(); }

private:
    std::variant<T, Error> data_;
};

// Specialisation for void results (just carries success/error)
template<>
class Result<void> {
public:
    Result()             : error_{} {}                               // success
    Result(Error error)  : error_(std::move(error)) {}               // NOLINT

    bool ok()    const noexcept { return error_.code == ErrorCode::Ok; }
    bool failed()const noexcept { return !ok(); }

    Error&       error()       { return error_; }
    const Error& error() const { return error_; }

private:
    Error error_;
};

using VoidResult = Result<void>;

// Convenience macro (like Rust's ?)
#define NN_TRY(expr)                        \
    do {                                    \
        auto _r = (expr);                   \
        if (_r.failed()) return _r.error(); \
    } while(0)

} // namespace nnstudio::core
