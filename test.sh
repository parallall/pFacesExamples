#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES="0_HelloWorld 1_VectorAdd 2_MatrixMul 3_ParallelReduction 4_Stencil2D_Heat 5_NBody 6_Histogram"

for ex in $EXAMPLES; do
  echo "============================================================"
  echo "Testing $ex"
  echo "============================================================"
  (cd "$SCRIPT_DIR/$ex" && sh ./test.sh)
done

echo "============================================================"
echo "All pFacesExamples passed!"
echo "============================================================"
