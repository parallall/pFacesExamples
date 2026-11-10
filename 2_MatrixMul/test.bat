@ECHO OFF
IF NOT EXIST .\kernel-pack\matmul.driver (
  call build.bat
  IF ERRORLEVEL 1 exit /b 1
)
pfaces -CG -d 1 -k matmul
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -t -p -k matmul -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
pfaces -CG -d 1 -p -hm -k matmul -cfg example.cfg
IF ERRORLEVEL 1 exit /b 1
