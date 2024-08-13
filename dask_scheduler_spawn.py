from dask.distributed import LocalCluster
from time import sleep

if __name__ == '__main__':
    local_client = LocalCluster(scheduler_port=8786)
    print(local_client.dashboard_link)
    while True:
        sleep(99999999)