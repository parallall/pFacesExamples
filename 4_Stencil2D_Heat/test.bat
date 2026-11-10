@ECHO OFF
IF NOT EXIST .\kernel-pack\heat2d.driver (
  call build.bat
  IF ERRORLEVEL 1 exit /b 1
)
pfaces -CG -d 1 -k heat2d
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -k heat2d -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -hm -k heat2d -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
