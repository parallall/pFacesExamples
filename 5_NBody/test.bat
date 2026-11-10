@ECHO OFF
IF NOT EXIST .\kernel-pack\nbody.driver (
  call build.bat
  IF ERRORLEVEL 1 exit /b 1
)
pfaces -CG -d 1 -k nbody
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -k nbody -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -hm -k nbody -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
