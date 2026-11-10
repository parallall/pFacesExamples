@ECHO OFF
IF NOT EXIST .\kernel-pack\reduction.driver (
  call build.bat
  IF ERRORLEVEL 1 exit /b 1
)
pfaces -CG -d 1 -k reduction
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -k reduction -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -hm -k reduction -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
