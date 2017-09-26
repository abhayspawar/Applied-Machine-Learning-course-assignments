mkdir task1
cd task1
git init

touch 1
git add 1
git commit -m "File 1 committed"
touch 2
git add 2
git commit -m "File 2 committed"
touch 3
git add 3
git commit -m "File 3 committed"
touch 4
git add 4
git commit -m "File 4 committed"
touch 5
git add 5
git commit -m "File 5 committed"

git branch feature master~4
git checkout feature

touch 6
git add 6
git commit -m "File 6 committed"
touch 7
git add 7
git commit -m "File 7 committed"
touch 8
git add 8
git commit -m "File 8 committed"

git rebase --onto feature master~2 master
git branch debug 'HEAD@{10}'
git checkout debug

touch 9
git add 9
git commit -m "File 9 committed"
git checkout -m feature~1
git add 7
git commit --amend -m "file 7 moved to commit 9"