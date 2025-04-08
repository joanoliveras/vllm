import os
import subprocess
import shlex
from subprocess import Popen
import time
from typing import List

class MultiServer:
    def __init__(self, server_args: str, output_path: str, num_servers: int, base_port: int = 8000):
        """
        Initialize multiple vLLM server instances.
        """
        self.server_args = server_args
        self.output_path = output_path
        self.num_servers = num_servers
        self.base_port = base_port
        self.servers = []
        self.processes = []
        
        # Create output directories if they don't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize server instances with different ports
        for i in range(num_servers):
            server_port = base_port + i
            server_specific_args = f"{server_args} --port {server_port}"
            server_out = open(os.path.join(self.output_path, f'server_{i}_out.log'), 'w')
            server_err = open(os.path.join(self.output_path, f'server_{i}_err.log'), 'w')
            
            self.servers.append({
                'id': i,
                'port': server_port,
                'args': server_specific_args,
                'out': server_out,
                'err': server_err
            })

    def run(self) -> list:
        """Launch all server instances and return their processes"""
        try:
            for server in self.servers:
                command = f'python3 -m vllm.entrypoints.openai.api_server {server["args"]}'
                process = subprocess.Popen(
                    shlex.split(command),
                    shell=False,
                    cwd='/',
                    stdout=server['out'],
                    stderr=server['err']
                )
                self.processes.append(process)
                time.sleep(1)
            
            return self.processes
        except Exception as e:
            print(f"Error starting servers: {e}")
            self.terminate(self.processes)
            raise e

    def terminate(self, processes: list) -> None:
        """Terminate all server processes and close log files"""
        for i, process in enumerate(processes):
            try:
                process.kill()
                process.terminate()
                process.wait()
            except Exception as e:
                print(f"Error terminating server {i}: {e}")
        
        # Close all log files
        for server in self.servers:
            if 'out' in server and server['out']:
                server['out'].close()
            if 'err' in server and server['err']:
                server['err'].close()
                
    def get_server_urls(self):
        """Return a list of server URLs"""
        return [f"http://127.0.0.1:{server['port']}" for server in self.servers]
