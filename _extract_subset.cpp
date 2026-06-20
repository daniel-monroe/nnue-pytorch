// Stream the first N entries of a binpack into a smaller binpack (same data, subset).
#include "lib/nnue_training_data_formats.h"
#include <cstdio>
#include <cstdlib>
using namespace binpack;
using namespace chess;

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: _extract_subset <in> <out> <N>\n"); return 1; }
    long N = atol(argv[3]);
    CompressedTrainingDataEntryReader r(argv[1], std::ios::binary | std::ios::in);
    CompressedTrainingDataEntryWriter w(argv[2], std::ios::binary | std::ios::trunc);
    long n = 0;
    while (r.hasNext() && n < N) {
        w.addTrainingDataEntry(r.next());
        if (++n % 2000000 == 0) fprintf(stderr, "  %ld\n", n);
    }
    fprintf(stderr, "wrote %ld entries to %s\n", n, argv[2]);
    return 0;
}
