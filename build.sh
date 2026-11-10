#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES="0_HelloWorld 1_VectorAdd 2_MatrixMul 3_ParallelReduction 4_Stencil2D_Heat 5_NBody 6_Histogram"

for ex in $EXAMPLES; do
  echo "=== Building $ex ==="
  (cd "$SCRIPT_DIR/$ex" && sh ./build.sh)
done

echo "All pFacesExamples built successfully."
