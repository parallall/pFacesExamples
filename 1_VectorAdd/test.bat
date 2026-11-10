@ECHO OFF
IF NOT EXIST .\kernel-pack\vecadd.driver (
  call build.bat
  IF ERRORLEVEL 1 exit /b 1
)
pfaces -CG -d 1 -k vecadd
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -k vecadd -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -hm -k vecadd -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
