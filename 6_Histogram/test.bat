@ECHO OFF
IF NOT EXIST .\kernel-pack\histogram.driver (
  call build.bat
  IF ERRORLEVEL 1 exit /b 1
)
pfaces -CG -d 1 -k histogram
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -k histogram -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -hm -k histogram -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
