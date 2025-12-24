@echo off
REM Script để push project lên GitHub cho Windows
REM Username: pl1201

echo 🚀 Bắt đầu push project lên GitHub...

REM Bước 1: Kiểm tra git đã được init chưa
if not exist ".git" (
    echo 📦 Khởi tạo git repository...
    git init
)

REM Bước 2: Thêm tất cả files
echo 📝 Thêm files vào git...
git add .

REM Bước 3: Commit
echo 💾 Tạo commit...
git commit -m "Initial commit: Question Answering System with ALBERT on SQuAD v1.1" -m "- Fine-tuning ALBERT-base for extractive QA" -m "- Robust preprocessing and post-processing" -m "- Early stopping to prevent overfitting" -m "- Results: EM 56.8%%, F1 70.8%%"

REM Bước 4: Đổi tên branch thành main
echo 🌿 Đổi tên branch thành main...
git branch -M main

REM Bước 5: Thêm remote
echo 🔗 Thêm remote repository...
echo ⚠️  LƯU Ý: Bạn cần tạo repo trên GitHub trước!
set /p repo_name="Nhập tên repo trên GitHub (hoặc Enter để dùng 'question-answering-system'): "
if "%repo_name%"=="" set repo_name=question-answering-system

REM Kiểm tra remote đã tồn tại chưa
git remote get-url origin >nul 2>&1
if %errorlevel% equ 0 (
    echo 🔄 Remote 'origin' đã tồn tại, cập nhật...
    git remote set-url origin https://github.com/pl1201/%repo_name%.git
) else (
    echo ➕ Thêm remote 'origin'...
    git remote add origin https://github.com/pl1201/%repo_name%.git
)

REM Bước 6: Push lên GitHub
echo ⬆️  Push lên GitHub...
git push -u origin main

echo ✅ Hoàn tất! Kiểm tra tại: https://github.com/pl1201/%repo_name%

pause

