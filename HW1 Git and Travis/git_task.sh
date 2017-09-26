
mkdir vbakshi		
cd vbakshi
git init

#Creating commits 1 to 5 and files 1 to 5
touch 1
git add 1
git commit -m "My commit number - 1"

touch 2
git add 2
git commit -m "My commit number - 2"

touch 3
git add 3
git commit -m "My commit number - 3"

touch 4
git add 4
git commit -m "My commit number - 4"

touch 5
git add 5
git commit -m "My commit number - 5"


git branch feature master~4
git checkout feature

#Creating commits 1 to 5 and files 6 to 8
touch 6
git add 6
git commit -m "My commit number - 6"

touch 7
git add 7
git commit -m "My commit number - 7"

touch 8
git add 8
git commit -m "My commit number - 8"


git rebase --onto feature master~2 master
git branch debug 'HEAD@{10}'
git checkout debug

touch 9

git add 9
git commit -m "My commit number 9"
git checkout feature~1 7
git add 7
git commit --amend -m 'ammend commit 9 by 7'
