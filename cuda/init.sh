#!/bin/sh

# bin, include, obj, srcの4つのディレクトリを作成する
mkdir -p bin
mkdir -p include
mkdir -p obj
mkdir -p src

# １つ前のディレクトリ名と同名のcuファイルをsrcディレクトリ内に作成する
touch ./src/$(basename $(dirname `readlink -f .`)).cu
# 実行権限の変更(読み書き実行可能)
chmod 777 ./bin ./include ./obj ./src
chmod 777 ./src/$(basename $(dirname `readlink -f .`)).cu