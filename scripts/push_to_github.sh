#!/bin/bash
# Script để push project lên GitHub
# Username: pl1201

echo "🚀 Bắt đầu push project lên GitHub..."

# Bước 1: Kiểm tra git đã được init chưa
if [ ! -d ".git" ]; then
    echo "📦 Khởi tạo git repository..."
    git init
fi

# Bước 2: Thêm tất cả files
echo "📝 Thêm files vào git..."
git add .

# Bước 3: Commit
echo "💾 Tạo commit..."
git commit -m "Initial commit: Question Answering System with ALBERT on SQuAD v1.1

- Fine-tuning ALBERT-base for extractive QA
- Robust preprocessing and post-processing
- Early stopping to prevent overfitting
- Results: EM 56.8%, F1 70.8%"

# Bước 4: Đổi tên branch thành main (GitHub standard)
echo "🌿 Đổi tên branch thành main..."
git branch -M main

# Bước 5: Thêm remote (người dùng cần thay <repo-name>)
echo "🔗 Thêm remote repository..."
echo "⚠️  LƯU Ý: Bạn cần tạo repo trên GitHub trước!"
echo "   Tên repo đề xuất: question-answering-system"
read -p "Nhập tên repo trên GitHub (hoặc Enter để dùng 'question-answering-system'): " repo_name
repo_name=${repo_name:-question-answering-system}

# Kiểm tra remote đã tồn tại chưa
if git remote get-url origin &>/dev/null; then
    echo "🔄 Remote 'origin' đã tồn tại, cập nhật..."
    git remote set-url origin https://github.com/pl1201/${repo_name}.git
else
    echo "➕ Thêm remote 'origin'..."
    git remote add origin https://github.com/pl1201/${repo_name}.git
fi

# Bước 6: Push lên GitHub
echo "⬆️  Push lên GitHub..."
git push -u origin main

echo "✅ Hoàn tất! Kiểm tra tại: https://github.com/pl1201/${repo_name}"

