
docker run -it -v $(pwd):$(pwd) -e HOME=$(pwd) -w $(pwd) -u $(id -u):$(id -g) --network=host dasf-seismic:cpu dask scheduler
    
#tcp://ra265679@143.106.16.194:8786 
#ssh ra265679@143.106.16.195
#ssh ra265679@143.106.16.196
#ssh ra265679@143.106.16.199
#ssh ra265679@143.106.16.195