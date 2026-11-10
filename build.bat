@echo off
setlocal
set EXAMPLES=0_HelloWorld 1_VectorAdd 2_MatrixMul 3_ParallelReduction 4_Stencil2D_Heat 5_NBody 6_Histogram

for %%E in (%EXAMPLES%) do (
  echo === Building %%E ===
  pushd "%~dp0%%E"
  call build.bat
  if errorlevel 1 (
    popd
    exit /b 1
  )
  popd
)

echo All pFacesExamples built successfully.
