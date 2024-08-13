from dask.distributed import Client
from time import sleep

if __name__ == '__main__':
    local_client = Client()
    while True:
        sleep(99999999)