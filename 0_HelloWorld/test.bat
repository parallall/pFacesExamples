@ECHO OFF
IF NOT EXIST .\kernel-pack\example.driver (
  call build.bat
  IF ERRORLEVEL 1 exit /b 1
)
pfaces -CG -d 1 -k example -v0
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -k example -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -hm -k example -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
