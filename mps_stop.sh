export CUDA_VISIBLE_DEVICES="0"
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0 -c DEFAULT
