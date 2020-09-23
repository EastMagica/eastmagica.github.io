# Blog with Hugo

`blog` 库用于同步源代码

`public` 用于发布页面

TODO: Themes 作为 master 分支的 submodule 存在, 相关修改不会提交(如 mathjax 文件中).

init githubpages

```bash
cd public
git init
git remote add origin git@github.com:eastmagica/eastmagica.github.io.git

git add --all
git commit -m "blog added"
git push -u origin master
```

usually,

```bash
(cd ..; hugo --theme=hugo_theme_robust)
git add --all
git commit -m "<some change message>"
git push -u origin master
```
