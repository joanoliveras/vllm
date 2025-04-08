import os
import shlex
import subprocess
from subprocess import Popen
from typing import List

class RouterServer:
    def __init__(self, output_path: str, router_port: int, server_ports: list, 
                 base_model: str, lora_adapters: list,router_script_path: str = "/Users/joanoliverastorra/Documents/IBM/production_stack/src/vllm_router/app.py"):
        """
        Initialize and run the vLLM router
        
        Args:
            output_path: Directory to save router logs
            router_port: Port for the router service
            server_ports: List of ports where vLLM servers are running
            base_model: Base model name
            lora_adapters: List of LoRA adapters loaded on the servers
        """
        self.output_path = output_path
        self.router_port = router_port
        self.server_ports = server_ports
        self.base_model = base_model
        self.lora_adapters = lora_adapters
        self.router_out = None
        self.router_err = None
        self.process = None
        self.router_script_path = router_script_path
        
    def run(self) -> Popen:
        try:
            self.router_out = open(os.path.join(self.output_path, 'router_out.log'), 'w')
            self.router_err = open(os.path.join(self.output_path, 'router_err.log'), 'w')
            
            # Build static backends list (each server duplicated for each model it serves)
            backends = []
            models = []
            
            # Add base model for each server
            for port in self.server_ports:
                backends.append(f"http://localhost:{port}")
                models.append(self.base_model)
            
            # Add LoRA adapters for each server
            for adapter in self.lora_adapters:
                for port in self.server_ports:
                    backends.append(f"http://localhost:{port}")
                    models.append(adapter)
            
            static_backends = ",".join(backends)
            static_models = ",".join(models)
            
            # Build router command with default settings
            command = (
                    f"python3 {self.router_script_path} --port {self.router_port} "
                    f"--service-discovery static "
                    f"--static-backends \"{static_backends}\" "
                    f"--static-models \"{static_models}\" "
                    f"--log-stats "
                    f"--log-stats-interval 10 "
                    f"--engine-stats-interval 10 "
                    f"--request-stats-window 10 "
                    f"--routing-logic roundrobin "
                )
            
            print(f"Starting router with command: {command}")
            
            process = subprocess.Popen(
                shlex.split(command),
                shell=False,
                cwd='/',
                stdout=self.router_out,
                stderr=self.router_err
            )
            
            self.process = process
            return process
            
        except Exception as e:
            print(f"Error starting router: {e}")
            if self.router_out:
                self.router_out.close()
            if self.router_err:
                self.router_err.close()
            raise e
            
    def terminate(self) -> None:
        if self.process:
            try:
                self.process.kill()
                self.process.terminate()
                self.process.wait()
            except Exception as e:
                print(f"Error terminating router: {e}")
        
        if self.router_out:
            self.router_out.close()
        if self.router_err:
            self.router_err.close()
