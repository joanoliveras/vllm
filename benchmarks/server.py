import subprocess
import shlex
import os
from subprocess import Popen


class Server:

    def __init__(self, server_args: str, output_path: str):
        super(Server, self).__init__()
        self.server_args = server_args
        self.output_path = output_path
        self.server_out = None
        self.server_err = None

    def run(self) -> Popen:
        try:
            self.server_out = open(os.path.join(self.output_path, 'server_out.log'), 'w')
            self.server_err = open(os.path.join(self.output_path, 'server_err.log'), 'w')
            command = f'python3 -m vllm.entrypoints.openai.api_server {self.server_args}'
            open_subprocess = subprocess.Popen(
                shlex.split(command),
                shell=False,
                cwd='/',
                stdout=self.server_out,
                stderr=self.server_err
            )
            return open_subprocess
        except Exception as e:
            print(e)
            if self.server_out:
                self.server_out.close()
            if self.server_err:
                self.server_err.close()
            raise e

    def terminate(self, open_subprocess: Popen) -> None:
        open_subprocess.kill()
        open_subprocess.terminate()
        open_subprocess.wait()
        if self.server_out:
            self.server_out.close()
        if self.server_err:
            self.server_err.close()
