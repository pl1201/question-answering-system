# 🚀 Hướng dẫn Push Project lên GitHub

## Bước 1: Tạo Repository trên GitHub

1. Đăng nhập vào GitHub: https://github.com/login
2. Click nút **"+"** ở góc trên bên phải → chọn **"New repository"**
3. Điền thông tin:
   - **Repository name**: `question-answering-system` (hoặc tên bạn muốn)
   - **Description**: `Fine-tuning ALBERT for Question Answering on SQuAD v1.1`
   - **Visibility**: Public (hoặc Private nếu muốn)
   - ⚠️ **KHÔNG** tích "Initialize with README" (vì đã có README rồi)
4. Click **"Create repository"**

## Bước 2: Push Code lên GitHub

### Cách 1: Dùng Script (Windows)

```bash
# Chạy script
scripts\push_to_github.bat
```

### Cách 2: Làm thủ công

```bash
# 1. Kiểm tra git status
git status

# 2. Thêm tất cả files
git add .

# 3. Tạo commit
git commit -m "Initial commit: Question Answering System with ALBERT on SQuAD v1.1"

# 4. Đổi tên branch thành main
git branch -M main

# 5. Thêm remote (thay <repo-name> bằng tên repo bạn đã tạo)
git remote add origin https://github.com/pl1201/question-answering-system.git

# 6. Push lên GitHub
git push -u origin main
```

### Cách 3: Dùng GitHub CLI (nếu đã cài)

```bash
# Cài GitHub CLI: https://cli.github.com/
gh repo create question-answering-system --public --source=. --remote=origin --push
```

## Bước 3: Kiểm tra

Sau khi push thành công, mở trình duyệt và vào:
```
https://github.com/pl1201/question-answering-system
```

## 🔐 Xác thực GitHub

Nếu gặp lỗi authentication:

### Option 1: Personal Access Token (Khuyến nghị)
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token → chọn quyền `repo`
3. Copy token và dùng khi push:
   ```bash
   git push -u origin main
   # Username: pl1201
   # Password: <paste-token-here>
   ```

### Option 2: SSH Key
1. Tạo SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Thêm vào GitHub: Settings → SSH and GPG keys → New SSH key
3. Đổi remote URL:
   ```bash
   git remote set-url origin git@github.com:pl1201/question-answering-system.git
   ```

## 📝 Lưu ý

- ✅ Đã có `.gitignore` để bỏ qua checkpoints, data, logs
- ✅ Không push file `.json` lớn (data)
- ✅ Không push model checkpoints (`.pt`, `.bin`)
- ✅ README.md đã được format đẹp với badges và emoji

## 🎯 Sau khi push thành công

1. ✅ Kiểm tra README hiển thị đúng trên GitHub
2. ✅ Thêm topics: `question-answering`, `albert`, `squad`, `nlp`, `pytorch`
3. ✅ Thêm description ngắn gọn
4. ✅ Enable GitHub Pages (nếu muốn)

## 🔄 Cập nhật sau này

```bash
git add .
git commit -m "Update: mô tả thay đổi"
git push
```

