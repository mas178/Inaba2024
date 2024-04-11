#!/bin/bash

# install Julia
wget https://julialang-s3.julialang.org/bin/linux/aarch64/1.10/julia-1.10.0-linux-aarch64.tar.gz
tar -xzf julia-1.10.0-linux-aarch64.tar.gz -C /opt
ln -s /opt/julia-1.10.0/bin/julia /usr/local/bin/julia
