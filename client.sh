docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu python3 train-model.py --data data/F3_train.zarr \
    --address tcp://143.106.16.194:8786 \
    --inline-window 2 --trace-window 2 --samples-window 2 \
    --attribute COS-INST-PHASE --iteration 1
