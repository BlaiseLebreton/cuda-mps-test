nvcc -o nbody  nbody.cu && echo "Compiled nbody"

nsys profile -o rep1 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ./nbody 150000 -5 &
nsys profile -o rep2 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ./nbody 150000  0
