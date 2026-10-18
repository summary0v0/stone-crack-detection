@echo off
setlocal
chcp 65001 >nul
title 石材裂纹检测 - 安装环境

if not exist "%~dp0setup_env.ps1" (
    echo 找不到 setup_env.ps1，请将本文件和项目文件放在同一个文件夹。
    pause
    exit /b 1
)

echo 正在安装本机运行环境，请保持联网并等待完成。
echo 首次安装需要下载约 2.7 GB 的 PyTorch 安装包。
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_env.ps1" %*
set "taskExitCode=%errorlevel%"
echo.
if "%taskExitCode%"=="0" (
    echo 环境安装成功。现在可以双击“启动石材裂纹检测.cmd”。
) else (
    echo 环境安装失败，请查看上方错误信息。退出代码：%taskExitCode%
)
pause
exit /b %taskExitCode%
