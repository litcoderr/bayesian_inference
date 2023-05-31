cd ..

container_name="bayesian_inference"
CUDA_VISIBLE_DEVICES='0'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --name $container_name \
    --mount src=$(pwd),dst=/$container_name,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -p 7777:8888 \
    -w /$container_name \
    $container_name \
    bash -c "bash" \
