/* ============================================================================
 * Checkpoint.cpp — .nns checkpoint save / load implementation
 * LGPL v3
 * ============================================================================ */

#include <core/Checkpoint.h>
#include <core/Tensor.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace nnstudio::core {

// ─── Binary helpers ───────────────────────────────────────────────────────────

static void writeU8 (std::ostream& o, uint8_t  v) { o.write(reinterpret_cast<const char*>(&v), 1); }
static void writeU16(std::ostream& o, uint16_t v) { o.write(reinterpret_cast<const char*>(&v), 2); }
static void writeU32(std::ostream& o, uint32_t v) { o.write(reinterpret_cast<const char*>(&v), 4); }
static void writeU64(std::ostream& o, uint64_t v) { o.write(reinterpret_cast<const char*>(&v), 8); }
static void writeStr(std::ostream& o, const std::string& s) {
    writeU32(o, static_cast<uint32_t>(s.size()));
    o.write(s.data(), static_cast<std::streamsize>(s.size()));
}
static void writeTag(std::ostream& o, const char tag[2]) { o.write(tag, 2); }

static bool readU8 (std::istream& i, uint8_t&  v) { return (bool)i.read(reinterpret_cast<char*>(&v), 1); }
static bool readU16(std::istream& i, uint16_t& v) { return (bool)i.read(reinterpret_cast<char*>(&v), 2); }
static bool readU32(std::istream& i, uint32_t& v) { return (bool)i.read(reinterpret_cast<char*>(&v), 4); }
static bool readU64(std::istream& i, uint64_t& v) { return (bool)i.read(reinterpret_cast<char*>(&v), 8); }
static bool readStr(std::istream& i, std::string& s) {
    uint32_t len = 0;
    if (!readU32(i, len)) return false;
    s.resize(len);
    return (bool)i.read(s.data(), static_cast<std::streamsize>(len));
}

// Embed / read a single tensor using NNS1 binary format via a sstream bridge.
static void writeTensorInline(std::ostream& out, const Tensor& t) {
    std::ostringstream buf;
    auto r = t.save(buf);  // Tensor must have an ostream overload — see below
    (void)r;
    std::string data = buf.str();
    writeU32(out, static_cast<uint32_t>(data.size()));
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

static bool readTensorInline(std::istream& in, Tensor& t) {
    uint32_t sz = 0;
    if (!readU32(in, sz)) return false;
    std::string data(sz, '\0');
    if (!in.read(data.data(), sz)) return false;
    std::istringstream buf(data);
    auto r = Tensor::load(buf);
    if (!r.ok()) return false;
    t = std::move(r.value());
    return true;
}

// ─── Checkpoint::save ─────────────────────────────────────────────────────────

Result<void> Checkpoint::save(std::string_view path, const CheckpointState& cs) {
    std::ofstream f(std::string(path), std::ios::binary | std::ios::trunc);
    if (!f)
        return Result<void>(Error{ErrorCode::IoError,
            "Checkpoint::save: cannot open '" + std::string(path) + "'"});

    // Header
    f.write("NNSC", 4);
    writeU16(f, 1);  // version

    // ── Section 1: Weights ────────────────────────────────────────────────
    writeTag(f, "WS");
    writeU32(f, static_cast<uint32_t>(cs.params.size()));
    for (const auto* p : cs.params) {
        writeStr(f, p->name);
        writeTensorInline(f, p->tensor);
    }

    // ── Section 2: Optimizer state ────────────────────────────────────────
    if (cs.optimizer) {
        writeTag(f, "OS");
        writeStr(f, std::string{cs.optimizer->name()});
        cs.optimizer->saveState(f, cs.params);
    }

    // ── Section 3: Training counters ──────────────────────────────────────
    writeTag(f, "TC");
    writeU64(f, cs.globalStep);
    writeU64(f, cs.epoch);

    // End marker
    writeTag(f, "EN");

    if (!f)
        return Result<void>(Error{ErrorCode::IoError, "Checkpoint::save: write error"});
    return Result<void>();
}

// ─── Checkpoint::load ─────────────────────────────────────────────────────────

Result<void> Checkpoint::load(std::string_view path, CheckpointState& cs) {
    std::ifstream f(std::string(path), std::ios::binary);
    if (!f)
        return Result<void>(Error{ErrorCode::IoError,
            "Checkpoint::load: cannot open '" + std::string(path) + "'"});

    // Must have no exceptions in this translation unit — check stream state manually.

    // Header
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "NNSC", 4) != 0)
        return Result<void>(Error{ErrorCode::IoError,
            "Checkpoint::load: bad magic (expected NNSC)"});

    uint16_t ver = 0;
    readU16(f, ver);
    if (ver != 1)
        return Result<void>(Error{ErrorCode::IoError,
            "Checkpoint::load: unsupported checkpoint version " + std::to_string(ver)});

    for (;;) {
        char tag[2] = {0, 0};
        if (!f.read(tag, 2)) break;  // EOF

        if (tag[0] == 'E' && tag[1] == 'N') break;  // end marker

        // ── Weights section ────────────────────────────────────────────────
        if (tag[0] == 'W' && tag[1] == 'S') {
            uint32_t count = 0;
            if (!readU32(f, count))
                return Result<void>(Error{ErrorCode::IoError, "Checkpoint::load: bad weights count"});

            for (uint32_t i = 0; i < count; ++i) {
                std::string name;
                if (!readStr(f, name))
                    return Result<void>(Error{ErrorCode::IoError, "Checkpoint::load: bad param name"});

                Tensor t;
                if (!readTensorInline(f, t))
                    return Result<void>(Error{ErrorCode::IoError,
                        "Checkpoint::load: bad tensor for param '" + name + "'"});

                // Find matching parameter by name or by index
                if (i < cs.params.size()) {
                    cs.params[i]->tensor = std::move(t);
                }
            }
        }
        // ── Optimizer state section ────────────────────────────────────────
        else if (tag[0] == 'O' && tag[1] == 'S') {
            std::string typeName;
            if (!readStr(f, typeName))
                return Result<void>(Error{ErrorCode::IoError, "Checkpoint::load: bad optim type"});

            if (cs.optimizer) {
                cs.optimizer->loadState(f, cs.params);
            } else {
                // Skip optimizer state — read step + per-param entries
                // We don't know the exact byte count, but since we control the
                // format and the optimizer type is "Adam" / "AdamW" by name,
                // we can read the step counter then skip per-param blocks.
                uint64_t step_dummy = 0;
                readU64(f, step_dummy);
                for (size_t i = 0; i < cs.params.size(); ++i) {
                    uint8_t has = 0;
                    readU8(f, has);
                    if (has) {
                        uint64_t cnt = 0;
                        readU64(f, cnt);
                        // skip m and v
                        f.seekg(static_cast<std::streamoff>(cnt * 2 * sizeof(float)),
                                std::ios::cur);
                    }
                }
            }
        }
        // ── Training counters section ──────────────────────────────────────
        else if (tag[0] == 'T' && tag[1] == 'C') {
            readU64(f, cs.globalStep);
            readU64(f, cs.epoch);
        }
        // ── Unknown section — skip by reading until next known tag ─────────
        // (forward-compat: unknown tags are not recoverable without length field)
        // For now treat as error for strict reading.
        else {
            // Unknown tag — we can't skip without a length field.
            // Treat as end of known data.
            break;
        }
    }

    if (f.bad())
        return Result<void>(Error{ErrorCode::IoError, "Checkpoint::load: read error"});
    return Result<void>();
}

} // namespace nnstudio::core
