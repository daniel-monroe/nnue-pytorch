#!/usr/bin/env bash
# Builds data_loader/cpp/build/libtraining_data_loader.so with g++ directly, using the same flags as
# the CMake Release build (for machines without cmake). No PGO.
#
# Usage: ./scripts/build_data_loader_gxx.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT_DIR/data_loader/cpp"
OUT="$SRC/build"
CXX=${CXX:-g++}
mkdir -p "$OUT"
FLAGS="-march=native -Wall -O3 -DNDEBUG -std=gnu++20 -fPIC -mbmi2 -DHAS_BMI2"
"$CXX" $FLAGS -c "$SRC/training_data_loader.cpp" -o "$OUT/training_data_loader.o" &
"$CXX" $FLAGS -c "$SRC/training_data_loader_abi.cpp" -o "$OUT/training_data_loader_abi.o" &
wait %1 && wait %2
"$CXX" -fPIC -march=native -O3 -shared -Wl,-soname,libtraining_data_loader.so \
  -o "$OUT/libtraining_data_loader.so.tmp" "$OUT/training_data_loader.o" "$OUT/training_data_loader_abi.o" -lpthread
mv "$OUT/libtraining_data_loader.so.tmp" "$OUT/libtraining_data_loader.so"
rm -f "$OUT/training_data_loader.o" "$OUT/training_data_loader_abi.o"
echo "built $OUT/libtraining_data_loader.so"
