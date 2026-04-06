/* =============================================================================
 * BpeTokenizerPlugin.cpp — NNStudio built-in BPE tokenizer reference plugin
 * MIT licensed (reference plugin — may be freely embedded in other projects)
 *
 * Implements NN_PLUGIN_TOKENIZER with a minimal byte-level BPE tokenizer.
 * A compact built-in merge table encodes common English text without any
 * external vocab file.  For production use, load a custom merge list via
 * the future NNTokenizerVTable::load_vocab() extension.
 *
 * @kb: ai-standards-kb/annexes/A06-Tokenization-Deep-Dive.md §BPE
 * ============================================================================ */

#include <plugin-api/nnstudio_plugin.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

/* ── BPE vocabulary ──────────────────────────────────────────────────────── */

// Each merge rule: (left_token, right_token) → merged_token
// Rank = index in this table (lower = applied earlier).
//
// Byte range 0-255 occupy token IDs 0-255.
// All merged tokens start at ID 256.
struct MergeRule { const char* left; const char* right; const char* merged; };

// clang-format off
static const MergeRule kMerges[] = {
    // space + letter merges (GPT-2 style Ġ represented as ' ')
    {" ", "t",  " t"},  // 256
    {" ", "a",  " a"},  // 257
    {" ", "o",  " o"},  // 258
    {" ", "i",  " i"},  // 259
    {" ", "s",  " s"},  // 260
    {" ", "n",  " n"},  // 261
    {" ", "w",  " w"},  // 262
    {" ", "h",  " h"},  // 263
    {" ", "b",  " b"},  // 264
    {" ", "f",  " f"},  // 265
    {" ", "c",  " c"},  // 266
    {" ", "r",  " r"},  // 267
    {" ", "p",  " p"},  // 268
    {" ", "m",  " m"},  // 269
    {" ", "l",  " l"},  // 270
    {" ", "d",  " d"},  // 271
    {" ", "e",  " e"},  // 272
    {" ", "g",  " g"},  // 273
    {" ", "u",  " u"},  // 274
    {" ", "y",  " y"},  // 275
    // common bigrams
    {"t", "h",  "th"},  // 276
    {"h", "e",  "he"},  // 277
    {"i", "n",  "in"},  // 278
    {"e", "r",  "er"},  // 279
    {"a", "n",  "an"},  // 280
    {"r", "e",  "re"},  // 281
    {"o", "n",  "on"},  // 282
    {"e", "n",  "en"},  // 283
    {"a", "t",  "at"},  // 284
    {"e", "s",  "es"},  // 285
    {"o", "r",  "or"},  // 286
    {"n", "t",  "nt"},  // 287
    {"i", "s",  "is"},  // 288
    {"a", "r",  "ar"},  // 289
    {"e", "d",  "ed"},  // 290
    {"o", "u",  "ou"},  // 291
    {"i", "t",  "it"},  // 292
    {"a", "l",  "al"},  // 293
    {"i", "o",  "io"},  // 294
    // common trigrams
    {"th", "e",  "the"}, // 295
    {"in", "g",  "ing"}, // 296
    {" t", "he", " the"},// 297
    {"er", "s",  "ers"}, // 298
    {"i",  "on", "ion"}, // 299
    {"at", "i",  "ati"}, // 300
    {"ati", "on", "ation"},// 301
    {"or", "e",  "ore"}, // 302
    {"an", "d",  "and"}, // 303
    {" ", "and", " and"},// 304
    {"re", "s",  "res"}, // 305
    {"al", "l",  "all"}, // 306
    {"th", "at", "that"},// 307
    {"e",  "d ", "ed "}, // 308
    {"ou", "t",  "out"}, // 309
    {"ar", "e",  "are"}, // 310
    {" a",  "re", " are"},// 311
    {"ou", "r",  "our"}, // 312
    {"he", "r",  "her"}, // 313
    {"hi", "s",  "his"}, // 314
    {"wi", "th", "with"},// 315
    // digits
    {"0",  "0",  "00"},  // 316
    {"1",  "0",  "10"},  // 317
    {"2",  "0",  "20"},  // 318
};
// clang-format on

static constexpr int kNumMerges = static_cast<int>(sizeof(kMerges) / sizeof(kMerges[0]));
// Byte tokens: IDs 0-255 (single chars).
// Merged tokens: IDs 256 + merge_rank.
static constexpr int kVocabBase = 256;
static constexpr int kVocabSize = kVocabBase + kNumMerges;

/* ── BpeState ────────────────────────────────────────────────────────────── */

struct BpeState {
    // ID → string
    std::vector<std::string> id2str;
    // string → ID
    std::unordered_map<std::string, int32_t> str2id;
    // merge lookup: (left_str + '\x01' + right_str) → merged_str
    std::unordered_map<std::string, std::string> mergeMap;

    BpeState() {
        id2str.reserve(kVocabSize);
        str2id.reserve(kVocabSize);

        // Byte tokens 0-255
        for (int i = 0; i < 256; ++i) {
            std::string s(1, static_cast<char>(i));
            id2str.push_back(s);
            str2id[s] = static_cast<int32_t>(i);
        }

        // Merged tokens
        for (int r = 0; r < kNumMerges; ++r) {
            id2str.push_back(kMerges[r].merged);
            str2id[kMerges[r].merged] = static_cast<int32_t>(kVocabBase + r);

            std::string key{kMerges[r].left};
            key += '\x01';
            key += kMerges[r].right;
            mergeMap[key] = kMerges[r].merged;
        }
    }

    // BPE encode a single word (already separated into byte-symbols)
    std::vector<std::string> bpeWord(std::vector<std::string> symbols) const {
        // Repeatedly find and apply the lowest-rank merge
        while (symbols.size() > 1) {
            int bestRank = INT32_MAX;
            size_t bestPos = SIZE_MAX;
            std::string bestMerged;

            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                std::string key = symbols[i] + '\x01' + symbols[i + 1];
                auto it = mergeMap.find(key);
                if (it == mergeMap.end()) continue;

                auto idIt = str2id.find(it->second);
                int rank = (idIt != str2id.end())
                           ? idIt->second - kVocabBase : INT32_MAX;
                if (rank < bestRank) {
                    bestRank   = rank;
                    bestPos    = i;
                    bestMerged = it->second;
                }
            }
            if (bestPos == SIZE_MAX) break;  // no more merges available
            symbols[bestPos] = bestMerged;
            symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(bestPos) + 1);
        }
        return symbols;
    }
};

/* ── Plugin lifecycle ────────────────────────────────────────────────────── */

static BpeState* castState(NNPluginHandle h) {
    return static_cast<BpeState*>(h);
}

static NNPluginHandle bpe_create(void) {
    return static_cast<NNPluginHandle>(new BpeState{});
}

static void bpe_destroy(NNPluginHandle h) {
    delete castState(h);
}

/* ── Tokenizer vtable implementation ─────────────────────────────────────── */

static void bpe_encode(NNPluginHandle h,
                       const char*   text,
                       int32_t**     ids_out,
                       int32_t*      count_out) {
    if (!h || !text || !ids_out || !count_out) return;
    auto* s = castState(h);

    // Split on word boundaries very simply (by space — reference plugin)
    // and apply BPE per word segment.
    std::string_view input(text);
    std::vector<std::string> allTokens;

    size_t start = 0;
    while (start < input.size()) {
        // Take a "word" as a sequence up to next space (keep the space prefix)
        size_t end = start;
        while (end < input.size() && input[end] != ' ') ++end;
        // Include trailing space if present
        if (end < input.size() && input[end] == ' ') ++end;

        if (end == start) { ++start; continue; }

        std::string_view word = input.substr(start, end - start);

        // Byte-level split
        std::vector<std::string> syms;
        syms.reserve(word.size());
        for (unsigned char c : word)
            syms.emplace_back(1, static_cast<char>(c));

        auto merged = s->bpeWord(std::move(syms));
        for (auto& tok : merged) allTokens.push_back(std::move(tok));
        start = end;
    }

    // Convert to IDs
    *count_out = static_cast<int32_t>(allTokens.size());
    *ids_out   = static_cast<int32_t*>(
        std::malloc(sizeof(int32_t) * allTokens.size()));
    for (size_t i = 0; i < allTokens.size(); ++i) {
        auto it = s->str2id.find(allTokens[i]);
        (*ids_out)[i] = (it != s->str2id.end())
            ? it->second
            : static_cast<int32_t>(static_cast<unsigned char>(allTokens[i][0]));
    }
}

static char* bpe_decode(NNPluginHandle h,
                        const int32_t* ids,
                        int32_t        count) {
    if (!h || !ids || count <= 0) {
        char* empty = static_cast<char*>(std::malloc(1));
        if (empty) empty[0] = '\0';
        return empty;
    }
    auto* s = castState(h);
    std::string result;
    result.reserve(static_cast<size_t>(count) * 3);
    for (int32_t i = 0; i < count; ++i) {
        int32_t id = ids[i];
        if (id >= 0 && id < static_cast<int32_t>(s->id2str.size()))
            result += s->id2str[static_cast<size_t>(id)];
        else
            result += '\xEF';  // replacement char start — unknown ID
    }
    char* out = static_cast<char*>(std::malloc(result.size() + 1));
    if (out) std::memcpy(out, result.c_str(), result.size() + 1);
    return out;
}

static void bpe_free_result(NNPluginHandle /*h*/, void* ptr) {
    std::free(ptr);
}

static uint32_t bpe_vocab_size(NNPluginHandle /*h*/) {
    return static_cast<uint32_t>(kVocabSize);
}

static const char* bpe_token_to_str(NNPluginHandle h, int32_t id) {
    auto* s = castState(h);
    if (id < 0 || id >= static_cast<int32_t>(s->id2str.size())) return nullptr;
    return s->id2str[static_cast<size_t>(id)].c_str();
}

static int32_t bpe_str_to_token(NNPluginHandle h, const char* token) {
    auto* s = castState(h);
    auto it = s->str2id.find(token);
    return (it != s->str2id.end()) ? it->second : -1;
}

static const char* bpe_doc_ref(NNPluginHandle /*h*/) {
    return "ai-standards-kb/annexes/A06-Tokenization-Deep-Dive.md#byte-pair-encoding";
}

static const NNTokenizerVTable kBpeVTable = {
    bpe_encode,
    bpe_decode,
    bpe_free_result,
    bpe_vocab_size,
    bpe_token_to_str,
    bpe_str_to_token,
    bpe_doc_ref,
};

/* ── Plugin descriptor ───────────────────────────────────────────────────── */

static const NNPluginDescriptor kDescriptor = {
    NNSTUDIO_PLUGIN_API_VERSION,
    NN_PLUGIN_TOKENIZER,
    "io.nnstudio.builtin.bpe-tokenizer",
    "BPE Tokenizer (built-in reference)",
    "0.1.0",
    "NNStudio Authors",
    "MIT",
    bpe_create,
    bpe_destroy,
    const_cast<NNTokenizerVTable*>(&kBpeVTable),
};

extern "C" const NNPluginDescriptor* nnstudio_plugin_descriptor(void) {
    return &kDescriptor;
}
