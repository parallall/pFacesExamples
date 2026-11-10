@echo off
setlocal
set EXAMPLES=0_HelloWorld 1_VectorAdd 2_MatrixMul 3_ParallelReduction 4_Stencil2D_Heat 5_NBody 6_Histogram

for %%E in (%EXAMPLES%) do (
  echo ============================================================
  echo Testing %%E
  echo ============================================================
  pushd "%~dp0%%E"
  call test.bat
  if errorlevel 1 (
    popd
    exit /b 1
  )
  popd
)

echo ============================================================
echo All pFacesExamples passed!
echo ============================================================
