/* ============================================================================
 * ComputeGraph.cpp — directed-acyclic-graph engine implementation
 * LGPL v3
 * ============================================================================ */

#include <core/ComputeGraph.h>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace nnstudio::core {

// ─── Graph construction ───────────────────────────────────────────────────────

NodeId ComputeGraph::input() {
    NodeId id = allocNode();          // will be 0 on first call
    nodeValues_[id] = Tensor{};       // placeholder; filled by forward()
    return id;
}

NodeId ComputeGraph::record(ILayer* layer, NodeId inputId) {
    assert(layer != nullptr);
    NodeId outId = allocNode();
    tape_.push_back({layer, inputId, outId});
    if (outputId_ == kInvalidNode) {
        outputId_ = outId;            // auto-track last node as default output
    } else {
        outputId_ = outId;            // keep updating so last record = output
    }
    return outId;
}

// ─── ILayer::build ────────────────────────────────────────────────────────────

Result<Shape> ComputeGraph::build(const Shape& inputShape) {
    if (tape_.empty()) {
        markBuilt();
        return Result<Shape>{inputShape};
    }

    Shape current = inputShape;
    for (auto& op : tape_) {
        auto r = op.layer->build(current);
        if (!r.ok()) return r;
        current = r.value();
    }
    markBuilt();
    return Result<Shape>{current};
}

// ─── ILayer::forward ─────────────────────────────────────────────────────────

Result<Tensor> ComputeGraph::forward(const Tensor& x, EvalTrace* /*trace*/) {
    if (tape_.empty()) return Result<Tensor>{x};

    if (traceMode_) traces_.clear();

    // The input node id is tape_[0].inputId (== 0 from input() call).
    // Store x into nodeValues_ for that id.
    NodeId inputNodeId = tape_.front().inputId;
    nodeValues_[inputNodeId] = x;

    for (auto& op : tape_) {
        const Tensor& in = nodeValues_[op.inputId];

        // Ensure layer is built
        if (!op.layer->isBuilt()) {
            Shape s  = in.shape();
            // Remove batch dim: build expects single-sample shape
            Shape ss(s.begin() + 1, s.end());
            if (ss.empty()) ss = s;
            auto br = op.layer->build(ss);
            if (!br.ok()) return Result<Tensor>{br.error()};
        }

        auto outR = op.layer->forward(in);
        if (!outR.ok()) return outR;

        nodeValues_[op.outputId] = outR.value();

        if (traceMode_) {
            EvalTrace tr;
            tr.layerName = op.layer->name();
            tr.layerType = std::string{op.layer->typeName()};
            tr.input     = in.clone();
            tr.output    = outR.value().clone();
            // gradOutput / gradInput will be filled in backward()
            traces_.push_back(std::move(tr));
        }
    }

    return Result<Tensor>{nodeValues_[outputId_]};
}

// ─── ILayer::backward ────────────────────────────────────────────────────────

Result<Tensor> ComputeGraph::backward(const Tensor& gradOut, EvalTrace* /*trace*/) {
    if (tape_.empty()) return Result<Tensor>{gradOut};

    // Seed: gradient of the last node's output
    std::unordered_map<NodeId, Tensor> gradMap;
    gradMap[outputId_] = gradOut;

    // Reverse traversal
    for (int i = static_cast<int>(tape_.size()) - 1; i >= 0; --i) {
        const auto& op        = tape_[static_cast<size_t>(i)];
        const Tensor& gradOfOut = gradMap[op.outputId];

        auto dInR = op.layer->backward(gradOfOut);
        if (!dInR.ok()) return dInR;

        // Accumulate (multi-path graphs would add here; single path = assign)
        auto it = gradMap.find(op.inputId);
        if (it != gradMap.end()) {
            // accumulate: grad += dInR  (for future multi-input support)
            // Phase 1 simplification: just overwrite (single path)
            it->second = dInR.value();
        } else {
            gradMap[op.inputId] = dInR.value();
        }

        if (traceMode_ && i < static_cast<int>(traces_.size())) {
            traces_[static_cast<size_t>(i)].gradOutput = gradOfOut.clone();
            traces_[static_cast<size_t>(i)].gradInput  = dInR.value().clone();
        }
    }

    // Return gradient w.r.t. graph input
    NodeId inputNodeId = tape_.front().inputId;
    auto it = gradMap.find(inputNodeId);
    if (it != gradMap.end()) return Result<Tensor>{it->second};
    return Result<Tensor>(Error{ErrorCode::InvalidArgument, "ComputeGraph::backward: no grad for input node"});
}

// ─── parameters ─────────────────────────────────────────────────────────────

std::vector<Parameter*> ComputeGraph::parameters() {
    std::vector<Parameter*> all;
    for (auto& op : tape_) {
        for (auto* p : op.layer->parameters()) {
            // de-duplicate (same layer might appear twice in theory)
            bool found = false;
            for (auto* q : all) { if (q == p) { found = true; break; } }
            if (!found) all.push_back(p);
        }
    }
    return all;
}

std::vector<const Parameter*> ComputeGraph::parameters() const {
    std::vector<const Parameter*> all;
    for (const auto& op : tape_) {
        for (const auto* p : op.layer->parameters()) {
            bool found = false;
            for (const auto* q : all) { if (q == p) { found = true; break; } }
            if (!found) all.push_back(p);
        }
    }
    return all;
}

// ─── setTraining ────────────────────────────────────────────────────────────

void ComputeGraph::setTraining(bool t) noexcept {
    training_ = t;
    for (auto& op : tape_) op.layer->setTraining(t);
}

} // namespace nnstudio::core

// ─── Global layer factory registry ───────────────────────────────────────────

namespace {

using FactoryMap = std::unordered_map<
    std::string,
    nnstudio::core::LayerConstructor>;

FactoryMap& factoryRegistry() {
    static FactoryMap s_registry;
    return s_registry;
}

// ── Minimal hand-written JSON helpers ────────────────────────────────────────

// Escape a string for JSON (handles backslash, quote, common control chars).
static std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;     break;
        }
    }
    return out;
}

// Skip whitespace in a JSON token stream (position = index into str).
static void skipWs(const std::string& s, size_t& p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' ||
                             s[p] == '\n' || s[p] == '\r')) ++p;
}

static bool expect(const std::string& s, size_t& p, char ch) {
    skipWs(s, p);
    if (p >= s.size() || s[p] != ch) return false;
    ++p; return true;
}

// Parse a JSON string value (advances p past the closing quote).
// Returns false on malformed input.
static bool parseString(const std::string& s, size_t& p, std::string& out) {
    skipWs(s, p);
    if (p >= s.size() || s[p] != '"') return false;
    ++p; // consume opening quote
    out.clear();
    while (p < s.size() && s[p] != '"') {
        if (s[p] == '\\') {
            ++p;
            if (p >= s.size()) return false;
            switch (s[p]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                default:   out += s[p]; break;
            }
        } else {
            out += s[p];
        }
        ++p;
    }
    if (p >= s.size()) return false;
    ++p; // consume closing quote
    return true;
}

// Parse a JSON unsigned integer (no leading zeros except "0" itself).
static bool parseUint(const std::string& s, size_t& p, uint32_t& out) {
    skipWs(s, p);
    if (p >= s.size() || !std::isdigit(static_cast<unsigned char>(s[p])))
        return false;
    out = 0;
    while (p < s.size() && std::isdigit(static_cast<unsigned char>(s[p]))) {
        out = out * 10 + static_cast<uint32_t>(s[p] - '0');
        ++p;
    }
    return true;
}

// Parse a JSON key:value pair where the key is a string and the value is
// also a string.  p should be positioned before the opening '"' of the key.
static bool parseStringPair(const std::string& s, size_t& p,
                             std::string& key, std::string& value) {
    if (!parseString(s, p, key)) return false;
    skipWs(s, p);
    if (p >= s.size() || s[p] != ':') return false;
    ++p;
    if (!parseString(s, p, value)) return false;
    return true;
}

// Parse {"key1":"val1","key2":"val2",...}
// Returns false if the opening brace is missing or any token is malformed.
static bool parseStringMap(const std::string& s, size_t& p,
                            std::unordered_map<std::string,std::string>& out) {
    if (!expect(s, p, '{')) return false;
    skipWs(s, p);
    if (p < s.size() && s[p] == '}') { ++p; return true; } // empty object
    for (;;) {
        skipWs(s, p);
        std::string k, v;
        if (!parseStringPair(s, p, k, v)) return false;
        out[k] = v;
        skipWs(s, p);
        if (p >= s.size()) return false;
        if (s[p] == '}') { ++p; return true; }
        if (s[p] != ',') return false;
        ++p;
    }
}

} // anonymous namespace

namespace nnstudio::core {

// ─── Static factory registry interface ───────────────────────────────────────

void ComputeGraph::registerLayerFactory(const std::string& type,
                                        LayerConstructor   ctor) {
    factoryRegistry()[type] = std::move(ctor);
}

std::vector<std::string> ComputeGraph::registeredLayerTypes() {
    std::vector<std::string> names;
    names.reserve(factoryRegistry().size());
    for (const auto& kv : factoryRegistry()) names.push_back(kv.first);
    return names;
}

// ─── toJson ──────────────────────────────────────────────────────────────────

std::string ComputeGraph::toJson() const {
    std::ostringstream o;
    o << "{\n";

    // outputNodeId
    o << "  \"outputNodeId\": " << outputId_ << ",\n";

    // nodes — one entry per OpRecord (outputId is the "node id")
    o << "  \"nodes\": [\n";
    for (size_t i = 0; i < tape_.size(); ++i) {
        const auto& op = tape_[i];
        o << "    {\n";
        o << "      \"id\": "   << op.outputId << ",\n";
        o << "      \"type\": \"" << jsonEscape(std::string{op.layer->typeName()}) << "\",\n";
        o << "      \"name\": \"" << jsonEscape(op.layer->name()) << "\",\n";
        o << "      \"config\": {";
        auto cfg = op.layer->config();
        bool first = true;
        for (const auto& kv : cfg) {
            if (!first) o << ", ";
            o << "\"" << jsonEscape(kv.first)  << "\": "
              << "\"" << jsonEscape(kv.second) << "\"";
            first = false;
        }
        o << "}\n";
        o << "    }";
        if (i + 1 < tape_.size()) o << ",";
        o << "\n";
    }
    o << "  ],\n";

    // edges — one per OpRecord: from inputId to outputId via layer index i
    o << "  \"edges\": [\n";
    for (size_t i = 0; i < tape_.size(); ++i) {
        const auto& op = tape_[i];
        o << "    {"
          << " \"from\": " << op.inputId
          << ", \"to\": "  << op.outputId
          << ", \"layer\": " << i
          << " }";
        if (i + 1 < tape_.size()) o << ",";
        o << "\n";
    }
    o << "  ]\n";
    o << "}";
    return o.str();
}

// ─── fromJson ────────────────────────────────────────────────────────────────

Result<ComputeGraph> ComputeGraph::fromJson(const std::string& json) {
    // ── 1. Parse root object ──────────────────────────────────────────────
    size_t p = 0;
    if (!expect(json, p, '{'))
        return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: expected '{'")};

    ComputeGraph graph;

    struct NodeDesc {
        uint32_t id{0};
        std::string type;
        std::string name;
        std::unordered_map<std::string, std::string> config;
    };
    struct EdgeDesc {
        uint32_t from{0}, to{0}, layerIdx{0};
    };

    std::vector<NodeDesc> nodes;
    std::vector<EdgeDesc> edges;
    uint32_t outputNodeId = kInvalidNode;

    // Parse top-level key-value pairs
    for (;;) {
        skipWs(json, p);
        if (p >= json.size()) break;
        if (json[p] == '}') { ++p; break; }

        std::string key;
        if (!parseString(json, p, key))
            return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: bad key")};
        skipWs(json, p);
        if (p >= json.size() || json[p] != ':')
            return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: expected ':'")};
        ++p;

        if (key == "outputNodeId") {
            if (!parseUint(json, p, outputNodeId))
                return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: bad outputNodeId")};
        }
        else if (key == "nodes") {
            // Parse array of node objects
            skipWs(json, p);
            if (!expect(json, p, '['))
                return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: expected '[' for nodes")};
            for (;;) {
                skipWs(json, p);
                if (p < json.size() && json[p] == ']') { ++p; break; }
                if (!expect(json, p, '{'))
                    return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: expected '{' for node")};
                NodeDesc nd;
                for (;;) {
                    skipWs(json, p);
                    if (p < json.size() && json[p] == '}') { ++p; break; }
                    std::string nk;
                    if (!parseString(json, p, nk))
                        return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: bad node key")};
                    skipWs(json, p);
                    if (p >= json.size() || json[p] != ':')
                        return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: expected ':' in node")};
                    ++p;
                    if (nk == "id") {
                        if (!parseUint(json, p, nd.id))
                            return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: bad node id")};
                    } else if (nk == "type") {
                        if (!parseString(json, p, nd.type))
                            return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: bad node type")};
                    } else if (nk == "name") {
                        if (!parseString(json, p, nd.name))
                            return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: bad node name")};
                    } else if (nk == "config") {
                        if (!parseStringMap(json, p, nd.config))
                            return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: bad config map")};
                    } else {
                        // skip unknown value (string or number only for now)
                        skipWs(json, p);
                        if (p < json.size() && json[p] == '"') {
                            std::string tmp; parseString(json, p, tmp);
                        } else {
                            while (p < json.size() && json[p] != ',' && json[p] != '}') ++p;
                        }
                    }
                    skipWs(json, p);
                    if (p < json.size() && json[p] == ',') ++p;
                }
                nodes.push_back(std::move(nd));
                skipWs(json, p);
                if (p < json.size() && json[p] == ',') ++p;
            }
        }
        else if (key == "edges") {
            skipWs(json, p);
            if (!expect(json, p, '['))
                return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: expected '[' for edges")};
            for (;;) {
                skipWs(json, p);
                if (p < json.size() && json[p] == ']') { ++p; break; }
                if (!expect(json, p, '{'))
                    return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: expected '{' for edge")};
                EdgeDesc ed;
                for (;;) {
                    skipWs(json, p);
                    if (p < json.size() && json[p] == '}') { ++p; break; }
                    std::string ek;
                    if (!parseString(json, p, ek))
                        return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: bad edge key")};
                    skipWs(json, p);
                    if (p >= json.size() || json[p] != ':')
                        return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: expected ':' in edge")};
                    ++p;
                    if      (ek == "from")  parseUint(json, p, ed.from);
                    else if (ek == "to")    parseUint(json, p, ed.to);
                    else if (ek == "layer") parseUint(json, p, ed.layerIdx);
                    else { while (p < json.size() && json[p] != ',' && json[p] != '}') ++p; }
                    skipWs(json, p);
                    if (p < json.size() && json[p] == ',') ++p;
                }
                edges.push_back(ed);
                skipWs(json, p);
                if (p < json.size() && json[p] == ',') ++p;
            }
        }
        else {
            // skip unknown top-level value
            skipWs(json, p);
            while (p < json.size() && json[p] != ',' && json[p] != '}') ++p;
        }

        skipWs(json, p);
        if (p < json.size() && json[p] == ',') ++p;
    }

    // ── 2. Instantiate layers via factory ────────────────────────────────
    const auto& reg = factoryRegistry();
    for (auto& nd : nodes) {
        auto it = reg.find(nd.type);
        if (it == reg.end())
            return Result<ComputeGraph>{err(ErrorCode::NotImplemented,
                "fromJson: no factory for layer type '" + nd.type + "'")};
        auto layer = it->second(nd.config, nd.name);
        if (!layer)
            return Result<ComputeGraph>{err(ErrorCode::IoError,
                "fromJson: factory returned null for '" + nd.type + "'")};
        graph.ownedLayers_.push_back(std::move(layer));
    }

    // ── 3. Rebuild tape from edges ────────────────────────────────────────
    // edges are ordered by layer index; ensure nextId_ is correct
    graph.nextId_ = 0;
    // Build a map of node-id → next available id
    // For now: trust sequential NodeIds from the JSON
    // Allocate all node ids first
    uint32_t maxId = 0;
    for (const auto& ed : edges) {
        if (ed.from > maxId) maxId = ed.from;
        if (ed.to   > maxId) maxId = ed.to;
    }
    graph.nextId_ = maxId + 1;

    // input node: from of the first edge
    if (!edges.empty()) {
        graph.nodeValues_[edges[0].from] = Tensor{};
    }

    // Sort edges by layer index and build tape
    std::vector<EdgeDesc> sortedEdges = edges;
    std::sort(sortedEdges.begin(), sortedEdges.end(),
              [](const EdgeDesc& a, const EdgeDesc& b){ return a.layerIdx < b.layerIdx; });

    for (const auto& ed : sortedEdges) {
        if (ed.layerIdx >= graph.ownedLayers_.size())
            return Result<ComputeGraph>{err(ErrorCode::IoError, "fromJson: edge layer index out of range")};
        ILayer* layer = graph.ownedLayers_[ed.layerIdx].get();
        graph.tape_.push_back({layer, ed.from, ed.to});
    }

    graph.outputId_ = outputNodeId;

    return Result<ComputeGraph>{std::move(graph)};
}

} // namespace nnstudio::core
