import argparse

from .classes import Camera
from .utils import parse_dict, array_unique_by_key
from .constants import DEFAULT_CAMERA_PORT, DEFAULT_CHANNEL

import multiprocessing as mp

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        action="append",
        type=parse_dict,
        required=True,
        help='''s
            Example: --data ip=0.0.0.0,user=test
            Props: 
                ip=<ip string>
                port=<port number | optional>
                user=<username | optional>
                password=<password | optional>,
                channel=<channel number | optional>s
        ''',
    )
    parser.add_argument(
        '--global-user',
        type=str,
        required=False,
        help='''
            Global username used in device auth
        '''
    )
    parser.add_argument(
        '--global-password',
        type=str,
        required=False,
        help='''
            Global password used in device auth
        '''
    )
    parser.add_argument(
        '--global-port',
        type=str,
        required=False,
        help='''
            Global port used in your devices
        '''
    )
    parser.add_argument(
        '--global-channel',
        type=str,
        required=False,
        help='''
            Global channel used to get video from devices
        '''
    )
    args = parser.parse_args()

    global_user = args.global_user
    global_password = args.global_password
    global_channel = args.global_channel if args.global_channel is not None else DEFAULT_CHANNEL
    global_port = args.global_port if args.global_port is not None else DEFAULT_CAMERA_PORT

    new_ip_data: list[dict[str, str]]  = [{
        "username": item.user if "user" in item else global_user,
        "password": item.password if "password" in item is not None else global_password,
        "port": str(item.port if "port" in item is not None else global_port),
        "channel": item.channel if "channel" in item is not None else global_channel,
        "ip": item['ip'],
    } for item in args.data]

    return array_unique_by_key(items=new_ip_data, key="ip")

def main():
    ip_data = get_args()
    processes = []
    for item in ip_data:
        if item['port'] is None or item['channel'] is None or item['ip'] is None or item['username'] is None or item['password'] is None:
            raise ValueError("One of the arguments contains None")
        process = mp.Process(target=Camera.worker, args=(item,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()