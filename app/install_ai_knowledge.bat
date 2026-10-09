@echo off
setlocal
cd /d %~dp0

if not exist "my_strategy\knowledge_base_system\sessions" mkdir "my_strategy\knowledge_base_system\sessions"
if not exist "my_strategy\knowledge_base_system\prompts" mkdir "my_strategy\knowledge_base_system\prompts"
if not exist "my_strategy\knowledge_base_system\dispatch_packets" mkdir "my_strategy\knowledge_base_system\dispatch_packets"
if not exist "my_strategy\knowledge_base\04_AI_Changes" mkdir "my_strategy\knowledge_base\04_AI_Changes"
if not exist "my_strategy\knowledge_base\05_AI_Sessions" mkdir "my_strategy\knowledge_base\05_AI_Sessions"
if not exist "my_strategy\knowledge_base\06_Experiments" mkdir "my_strategy\knowledge_base\06_Experiments"
if not exist "my_strategy\knowledge_base\07_Backtests" mkdir "my_strategy\knowledge_base\07_Backtests"
if not exist "my_strategy\knowledge_base\08_Data_Lineage" mkdir "my_strategy\knowledge_base\08_Data_Lineage"
if not exist "my_strategy\knowledge_base\09_Decisions" mkdir "my_strategy\knowledge_base\09_Decisions"
if not exist "my_strategy\knowledge_base\10_Commits" mkdir "my_strategy\knowledge_base\10_Commits"

python "my_strategy\scripts\install_git_hooks.py"
if errorlevel 1 (
  echo [WARN] Git hook installation failed. The knowledge base is still usable.
)

python "my_strategy\scripts\rebuild_obsidian_indexes.py"
if errorlevel 1 exit /b 1

python -m py_compile ^
  "my_strategy\scripts\ai_change_logger.py" ^
  "my_strategy\scripts\agent_dispatch.py" ^
  "my_strategy\scripts\install_git_hooks.py" ^
  "my_strategy\scripts\rebuild_obsidian_indexes.py" ^
  "my_strategy\scripts\post_commit_note.py"
if errorlevel 1 exit /b 1

echo.
echo KHQuant AI knowledge system initialized.
echo Open this folder as an Obsidian vault:
echo %CD%\my_strategy\knowledge_base
endlocal
