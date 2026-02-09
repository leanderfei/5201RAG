VSCode 集成终端推送

git status  # 列出所有已修改、未暂存的文件
git add .  #暂存所有更改
git add main.py #暂存指定文件（如只推送main.py）
git commit -m "修复XX功能bug：优化数据校验逻辑"
git pull origin main  # （推荐）拉取远程最新代码（避免冲突）,main是默认分支，若用master则替换
git push origin main  # 推送到 GitHub,首次推送后，后续可直接输git push