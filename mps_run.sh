nvcc -o nbody  nbody.cu && echo "Compiled nbody"

## TEST1
rm test1 -rf
mkdir test1
cd test1

# MPS OFF
../mps_stop.sh
mkdir mps_off && cd mps_off
nsys profile -o rep1 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 150000 10 0 0 &
nsys profile -o rep2 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 150000 10 0 0
wait
cd ..

# MPS ON
../mps_start.sh
mkdir mps_on && cd mps_on
nsys profile -o rep1 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 150000 10 0 0 &
nsys profile -o rep2 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 150000 10 0 0
wait
cd ..

# MPS ON & PRIO
mkdir mps_on_prio && cd mps_on_prio
nsys profile -o rep1 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 150000 10 0 -5 &
nsys profile -o rep2 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 150000 10 0  0
wait
cd ..
cd ..

## TEST2
rm test2 -rf
mkdir test2 && cd test2

# MPS OFF
../mps_stop.sh
mkdir mps_off && cd mps_off
nsys profile -o rep1 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 400000 4  50 0 &
nsys profile -o rep2 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 75000 20 100 0
wait
cd ..

# MPS ON
../mps_start.sh
mkdir mps_on && cd mps_on
nsys profile -o rep1 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 400000 4  50 0 &
nsys profile -o rep2 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 75000 20 100 0
wait
cd ..

# MPS ON & PRIO
mkdir mps_on_prio && cd mps_on_prio
nsys profile -o rep1 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 400000 4  50  0 &
nsys profile -o rep2 -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -f true -x true ../../nbody 75000 20 100 -5
wait
cd ..
cd ..
