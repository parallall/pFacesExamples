@ECHO OFF
set BUILDTYPE=Release
set CLEAN_BUILD=True

IF "%CLEAN_BUILD%" == "True" (
  IF EXIST .\build rmdir /S/Q .\build
  IF EXIST .\kernel-pack\histogram.driver del /Q .\kernel-pack\histogram.driver
)

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=%BUILDTYPE%
cmake --build . --config %BUILDTYPE%
cd ..
