// Multi-threaded pawn-structure counter over a directory of binpack files.
// Splits files round-robin across threads; each thread counts into its own
// mutex-guarded maps. A reporter thread snapshots every REPORT_SECS seconds,
// rewrites the full output file AND prints a compact top-K block to stdout so
// progress is visible live. Canonicalized by "flip sides" (color-swap = bswap64).
//   FULL (a-h), LEFT (a-d), RIGHT (e-h)
#include "lib/nnue_training_data_formats.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>

using namespace binpack;
using namespace chess;
namespace fs = std::filesystem;

struct Key { uint64_t a, b; bool operator==(const Key& o) const { return a==o.a && b==o.b; } };
struct KeyHash { size_t operator()(const Key& k) const {
    uint64_t h = k.a * 0x9E3779B97F4A7C15ull; h ^= k.b + 0x9E3779B97F4A7C15ull + (h<<6) + (h>>2); return (size_t)h; } };
using Map = std::unordered_map<Key, uint64_t, KeyHash>;

static inline Key canon(uint64_t wp, uint64_t bp) {
    uint64_t fw = __builtin_bswap64(bp), fb = __builtin_bswap64(wp);
    Key k1{wp,bp}, k2{fw,fb};
    return (k2.a < k1.a || (k2.a==k1.a && k2.b<k1.b)) ? k2 : k1;
}

static const uint64_t LEFT = 0x0F0F0F0F0F0F0F0Full, RIGHT = 0xF0F0F0F0F0F0F0F0ull;

struct Result {
    std::mutex mtx;
    Map full, left, right;
    long n = 0;
};

static std::atomic<bool> g_done{false};

static void worker(const std::vector<std::string>& files, long cap, Result* R) {
    for (const auto& f : files) {
        if (R->n >= cap) break;
        try {
            CompressedTrainingDataEntryReader r(f, std::ios::binary | std::ios::in);
            while (r.hasNext() && R->n < cap) {
                auto e = r.next();
                uint64_t wp = 0, bp = 0;
                for (Square s : e.pos.piecesBB(whitePawn)) wp |= 1ull << (int)s;
                for (Square s : e.pos.piecesBB(blackPawn)) bp |= 1ull << (int)s;
                Key kf = canon(wp, bp), kl = canon(wp & LEFT, bp & LEFT), kr = canon(wp & RIGHT, bp & RIGHT);
                {
                    std::lock_guard<std::mutex> lk(R->mtx);  // uncontended except at snapshot
                    R->full[kf]++; R->left[kl]++; R->right[kr]++; R->n++;
                }
            }
        } catch (...) { /* skip a bad/truncated file */ }
    }
}

static void merge(Map& dst, const Map& src) { for (auto& kv : src) dst[kv.first] += kv.second; }

// Snapshot all per-thread maps into G.* and return total positions.
static long snapshot(std::vector<Result>& res, Map& full, Map& left, Map& right) {
    long total = 0;
    full.clear(); left.clear(); right.clear();
    for (auto& r : res) {
        std::lock_guard<std::mutex> lk(r.mtx);
        merge(full, r.full); merge(left, r.left); merge(right, r.right); total += r.n;
    }
    return total;
}

// compact one-line encoding: ranks 8..1, files f0..f1, P=one side p=other, '/' between ranks
static std::string fen_pawns(uint64_t wp, uint64_t bp, int f0, int f1) {
    std::string s;
    for (int r = 7; r >= 0; --r) {
        for (int f = f0; f <= f1; ++f) { int b = r*8+f; s += (wp>>b&1) ? 'P' : (bp>>b&1) ? 'p' : '.'; }
        if (r > 0) s += '/';
    }
    return s;
}

// Build a frequency-sorted vector once; the two dumpers share it.
static std::vector<std::pair<Key,uint64_t>> sorted_entries(const Map& m) {
    std::vector<std::pair<Key,uint64_t>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](auto&x, auto&y){ return x.second>y.second; });
    return v;
}

// complete list: EVERY distinct structure, sorted by count desc, compact one-line form.
// Built in a single in-memory buffer then one fwrite (per-char stdio locking is far too
// slow for millions of structures).
static void dump_all_compact(const std::string& path, const std::vector<std::pair<Key,uint64_t>>& v,
                             size_t distinct, long total, int f0, int f1) {
    std::string buf;
    buf.reserve(v.size() * 80 + 256);
    char line[160];
    snprintf(line, sizeof line, "# %zu distinct pawn structures over %ld positions\n", distinct, total);
    buf += line;
    buf += "# count\tpct\tpawns (rank8..rank1, P=one side, p=mirror side)\n";
    for (auto& kv : v) {
        int n = snprintf(line, sizeof line, "%llu\t%.5f\t", (unsigned long long)kv.second,
                         100.0*(double)kv.second/(double)total);
        buf.append(line, n);
        buf += fen_pawns(kv.first.a, kv.first.b, f0, f1);
        buf += '\n';
    }
    FILE* o = fopen(path.c_str(), "wb");
    if (!o) return;
    fwrite(buf.data(), 1, buf.size(), o);
    fclose(o);
}

// complete list rendered as labelled chessboards (rank numbers + file letters), buffered.
static void dump_all_boards(const std::string& path, const std::vector<std::pair<Key,uint64_t>>& v,
                            size_t distinct, long total, int f0, int f1) {
    std::string buf;
    buf.reserve(v.size() * (size_t)(12 * (f1 - f0 + 3)) + 256);
    char line[160];
    snprintf(line, sizeof line, "# %zu distinct pawn structures over %ld positions, sorted by frequency\n", distinct, total);
    buf += line;
    buf += "# P = one side, p = mirror side (sides folded together by color-swap)\n";
    long idx = 0;
    for (auto& kv : v) {
        uint64_t wp = kv.first.a, bp = kv.first.b;
        int n = snprintf(line, sizeof line, "\n#%ld  count=%llu  %.5f%%\n", ++idx,
                         (unsigned long long)kv.second, 100.0*(double)kv.second/(double)total);
        buf.append(line, n);
        for (int r = 7; r >= 0; --r) {
            buf += char('1' + r); buf += "  ";
            for (int f = f0; f <= f1; ++f) { int b = r*8+f;
                buf += (wp>>b&1) ? 'P' : (bp>>b&1) ? 'p' : '.'; buf += ' '; }
            buf += '\n';
        }
        buf += "   ";
        for (int f = f0; f <= f1; ++f) { buf += char('a' + f); buf += ' '; }
        buf += '\n';
    }
    FILE* o = fopen(path.c_str(), "wb");
    if (!o) return;
    fwrite(buf.data(), 1, buf.size(), o);
    fclose(o);
}

static std::string render(uint64_t wp, uint64_t bp, int f0, int f1, const char* pad) {
    std::string s;
    for (int r = 7; r >= 0; --r) { s += pad;
        for (int f = f0; f <= f1; ++f) { int b = r*8+f;
            s += (wp>>b&1) ? 'P' : (bp>>b&1) ? 'p' : '.'; s += ' '; }
        s += '\n'; }
    return s;
}

// full detail -> file
static void dump_file(FILE* out, const char* name, const Map& m, long total, int topN, int f0, int f1) {
    std::vector<std::pair<Key,uint64_t>> v(m.begin(), m.end());
    int n = std::min((int)v.size(), topN);
    std::partial_sort(v.begin(), v.begin()+n, v.end(), [](auto&x, auto&y){ return x.second>y.second; });
    fprintf(out, "\n================ %s : %zu distinct structures ================\n", name, m.size());
    for (int i = 0; i < n; ++i) {
        fprintf(out, "\n#%d  count=%llu  (%.3f%% of positions)\n", i+1,
                (unsigned long long)v[i].second, 100.0*(double)v[i].second/(double)total);
        fputs(render(v[i].first.a, v[i].first.b, f0, f1, "    ").c_str(), out);
    }
}

// cumulative coverage of the top 10/100/1000/10000 structures
static void coverage(FILE* o, const char* name, const Map& m, long total) {
    std::vector<uint64_t> c; c.reserve(m.size());
    for (auto& kv : m) c.push_back(kv.second);
    int cap = std::min((size_t)10000, c.size());
    std::partial_sort(c.begin(), c.begin()+cap, c.end(), std::greater<uint64_t>());
    int cuts[] = {10, 100, 1000, 10000};
    fprintf(o, "%s: distinct=%zu\n", name, m.size());
    for (int cut : cuts) {
        int k = std::min((int)c.size(), cut);
        long double s = 0; for (int i = 0; i < k; ++i) s += (long double)c[i];
        fprintf(o, "   top %-5d : %6.2f%%  (k=%d)\n", cut, (double)(100.0L*s/(long double)total), k);
    }
}

// compact top-K -> stdout (for live monitoring)
static void dump_stdout(const char* name, const Map& m, long total, int topK, int f0, int f1) {
    std::vector<std::pair<Key,uint64_t>> v(m.begin(), m.end());
    int n = std::min((int)v.size(), topK);
    std::partial_sort(v.begin(), v.begin()+n, v.end(), [](auto&x, auto&y){ return x.second>y.second; });
    printf("--- %s (%zu distinct) top %d ---\n", name, m.size(), n);
    for (int i = 0; i < n; ++i) {
        printf("  #%d  %.3f%%  (n=%llu)\n", i+1, 100.0*(double)v[i].second/(double)total,
               (unsigned long long)v[i].second);
        fputs(render(v[i].first.a, v[i].first.b, f0, f1, "      ").c_str(), stdout);
    }
}

// Light summary (coverage + top-N ASCII). Cheap: written every reporting cycle.
static void write_light_file(const std::string& outpath, long total, int nfiles, int T,
                             const std::string& dir, const Map& full, const Map& left,
                             const Map& right, int topN) {
    FILE* out = fopen(outpath.c_str(), "w");
    if (!out) return;
    fprintf(out, "Pawn structures over %ld positions (%d files, %d threads) from %s\n", total, nfiles, T, dir.c_str());
    fprintf(out, "Canonicalized by flipping sides (color-swap). P = one side, p = other.\n");
    fprintf(out, "\n=== CUMULATIVE COVERAGE (top N structures as %% of positions) ===\n");
    coverage(out, "FULL  (a-h)", full, total);
    coverage(out, "LEFT  (a-d)", left, total);
    coverage(out, "RIGHT (e-h)", right, total);
    dump_file(out, "FULL BOARD (files a-h)", full, total, topN, 0, 7);
    dump_file(out, "LEFT SIDE (files a-d)", left, total, topN, 0, 3);
    dump_file(out, "RIGHT SIDE (files e-h)", right, total, topN, 4, 7);
    fclose(out);
}

// Heavy complete lists (every distinct structure): boards + compact vocab.
// Hundreds of MB; written ONCE at the end, not every cycle.
static void write_heavy_files(const std::string& outpath, long total,
                              const Map& full, const Map& left, const Map& right) {
    std::string base = outpath;
    size_t dot = base.rfind(".txt");
    if (dot != std::string::npos) base = base.substr(0, dot);
    auto vf = sorted_entries(full), vl = sorted_entries(left), vr = sorted_entries(right);
    // human-readable chessboards
    dump_all_boards(base + "_full_all.txt", vf, full.size(), total, 0, 7);
    dump_all_boards(base + "_left_all.txt", vl, left.size(), total, 0, 3);
    dump_all_boards(base + "_right_all.txt", vr, right.size(), total, 4, 7);
    // compact machine-readable vocab (count\tpct\tfen) for feature index mapping
    dump_all_compact(base + "_full_vocab.txt", vf, full.size(), total, 0, 7);
    dump_all_compact(base + "_left_vocab.txt", vl, left.size(), total, 0, 3);
    dump_all_compact(base + "_right_vocab.txt", vr, right.size(), total, 4, 7);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: pawn_structs_mt <dir> [capPerThread] [topN] [outfile] [threads] [reportSecs]\n"); return 1; }
    std::string dir = argv[1];
    long cap = argc > 2 ? atol(argv[2]) : 100000000;
    int topN = argc > 3 ? atoi(argv[3]) : 50;
    std::string outpath = argc > 4 ? argv[4] : "out_attr/pawn_structures.txt";
    int T = argc > 5 ? atoi(argv[5]) : (int)std::thread::hardware_concurrency();
    int reportSecs = argc > 6 ? atoi(argv[6]) : 30;
    if (T < 1) T = 1;

    std::vector<std::string> files;
    for (auto& p : fs::recursive_directory_iterator(dir))
        if (p.is_regular_file() && p.path().extension() == ".binpack")
            files.push_back(p.path().string());
    std::sort(files.begin(), files.end());
    fprintf(stderr, "found %zu binpack files; %d threads; cap %ld pos/thread; report every %ds\n",
            files.size(), T, cap, reportSecs);
    fflush(stderr);

    std::vector<std::vector<std::string>> buckets(T);
    for (size_t i = 0; i < files.size(); ++i) buckets[i % T].push_back(files[i]);

    std::vector<Result> res(T);
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back(worker, std::cref(buckets[t]), cap, &res[t]);

    // Reporter thread: every reportSecs, snapshot + write file + print compact block.
    std::thread reporter([&]() {
        Map full, left, right;
        int cycle = 0;
        while (!g_done.load()) {
            for (int s = 0; s < reportSecs && !g_done.load(); ++s)
                std::this_thread::sleep_for(std::chrono::seconds(1));
            long total = snapshot(res, full, left, right);
            if (total == 0) continue;
            write_light_file(outpath, total, (int)files.size(), T, dir, full, left, right, topN);
            printf("\n===REPORT cycle=%d positions=%ld===\n", ++cycle, total);
            printf("--- cumulative coverage (top N structures as %% of positions) ---\n");
            coverage(stdout, "FULL  (a-h)", full, total);
            coverage(stdout, "LEFT  (a-d)", left, total);
            coverage(stdout, "RIGHT (e-h)", right, total);
            printf("--- current top structures ---\n");
            dump_stdout("FULL BOARD (a-h)", full, total, 3, 0, 7);
            dump_stdout("LEFT (a-d)", left, total, 3, 0, 3);
            dump_stdout("RIGHT (e-h)", right, total, 3, 4, 7);
            fflush(stdout);
        }
    });

    for (auto& x : th) x.join();
    g_done.store(true);
    reporter.join();

    // Final authoritative dump: light summary + heavy complete board/vocab files.
    Map full, left, right;
    long total = snapshot(res, full, left, right);
    write_light_file(outpath, total, (int)files.size(), T, dir, full, left, right, topN);
    fprintf(stderr, "processed %ld positions total; writing complete board/vocab files...\n", total);
    fflush(stderr);
    write_heavy_files(outpath, total, full, left, right);
    fprintf(stderr, "wrote %s (+ _all/_vocab files)\n", outpath.c_str());
    fflush(stderr);
    return 0;
}
