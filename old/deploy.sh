#!/bin/bash

# IPアドレスをコマンドライン引数から取得
IP_ADDRESS=$1

# (ローカル) 必要なディレクトリを作成
mkdir -p Inaba2024/output Inaba2024/log Inaba2024/src

# (ローカル) 必要なファイルをコピー
cp src/EntryPoint.jl Inaba2024/src/EntryPoint.jl
cp src/Network.jl Inaba2024/src/Network.jl
cp src/Simulation.jl Inaba2024/src/Simulation.jl
cp Manifest.toml Inaba2024/Manifest.toml
## Project.tomlの最初の空行までをコピー
awk '/^$/ {exit} {print}' Project.toml > Inaba2024/Project.toml

# (ローカル) Inaba2024ディレクトリを圧縮
tar -czvf Inaba2024.tar.gz Inaba2024

# (ローカル -> リモート) Inaba2024.tar.gzをEC2インスタンスに転送
scp -i ~/.ssh/MyLabEC2.pem Inaba2024.tar.gz ec2-user@$IP_ADDRESS:/home/ec2-user

# (リモート) Inaba2024.tar.gzを解凍
echo "Extracting Inaba2024.tar.gz"
ssh -i ~/.ssh/MyLabEC2.pem ec2-user@$IP_ADDRESS "tar -xzvf /home/ec2-user/Inaba2024.tar.gz -C /home/ec2-user"

# ゴミファイルを削除
echo "Clean up"
ssh -i ~/.ssh/MyLabEC2.pem ec2-user@$IP_ADDRESS "find /home/ec2-user/Inaba2024 -name '._*' -delete"
ssh -i ~/.ssh/MyLabEC2.pem ec2-user@$IP_ADDRESS "rm -f /home/ec2-user/Inaba2024.tar.gz"
rm -rf Inaba2024
rm -f Inaba2024.tar.gz

# (リモート) Juliaのパッケージをインストール
echo "$(date) Starting to install Julia packages..."
ssh -i ~/.ssh/MyLabEC2.pem ec2-user@$IP_ADDRESS "cd Inaba2024 && julia --project=. -e 'using Pkg; Pkg.resolve()'"
ssh -i ~/.ssh/MyLabEC2.pem ec2-user@$IP_ADDRESS "cd Inaba2024 && julia --project=. -e 'using Pkg; Pkg.instantiate()'"
echo "$(date) Finished installing Julia packages."
