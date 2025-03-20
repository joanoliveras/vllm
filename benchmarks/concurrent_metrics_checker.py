import os
import re
import signal
import sys
import time
from multiprocessing import Process
from typing import List

import numpy as np
import requests

DEBUG_NOT_FINISHING_REQUESTS = False


class ConcurrentMetricsChecker(Process):

    def __init__(self, output_path: str, metrics_api_url: str, users: List[str], available_loras: List[str]):
        super(ConcurrentMetricsChecker, self).__init__()
        self.output_path = output_path
        self.metrics_api_url = metrics_api_url
        self.users = users
        self.available_loras = available_loras
        signal.signal(signal.SIGTERM, self.__signal_term_handler)

        self.time = []
        self.gpu_cache_usage_perc = []
        self.num_running = []
        self.num_waiting = []
        self.num_preempted = []
        self.num_preempted_workload = []
        self.finished = []
        self.processed_tokens_prompt = []
        self.processed_tokens_generation = []
        self.running_by_adapter = {}
        self.waiting_by_adapter = {}
        for index in range(len(self.available_loras) + 1):
            self.running_by_adapter[index] = []
            self.waiting_by_adapter[index] = []

        '''
        self.vtc_cost_by_user = {}
        self.vtc_waiting_by_user = {}
        self.vtc_running_by_user = {}
        for user in users:
            self.vtc_cost_by_user[user] = []
            self.vtc_waiting_by_user[user] = []
            self.vtc_running_by_user[user] = []

        self.polling_waiting_by_queue = {}
        self.polling_running_by_queue = {}
        for lora in self.available_loras:
            self.polling_waiting_by_queue[lora] = []
            self.polling_running_by_queue[lora] = []
        '''

    def run(self):
        def find_prometheus_metric_value(label: str, metrics_response: str):
            try:
                pattern = f"vllm:{label}(.+?) ([+-]?([0-9]*[.])?[0-9]+)\n"
                value = re.search(pattern, metrics_response).group(2)
            except:
                # print(f'Pattern error with label {label}')
                return 0
            return float(value)

        start_time = time.perf_counter()
        while True:
            if DEBUG_NOT_FINISHING_REQUESTS and (time.perf_counter() - start_time) > 300:
                self.__save_metrics()
            try:
                self.time.append(time.perf_counter() - start_time)
                metrics_response = requests.get(self.metrics_api_url).text

                self.gpu_cache_usage_perc.append(
                    find_prometheus_metric_value(
                        f"gpu_cache_usage_perc",
                        metrics_response
                    )
                )

                self.num_running.append(
                    find_prometheus_metric_value(
                        f"num_requests_running",
                        metrics_response
                    )
                )

                self.num_waiting.append(
                    find_prometheus_metric_value(
                        f"num_requests_waiting",
                        metrics_response
                    )
                )

                self.num_preempted.append(
                    find_prometheus_metric_value(
                        f"num_preemptions_total",
                        metrics_response
                    )
                )

                self.num_preempted_workload.append(
                    find_prometheus_metric_value(
                        f"num_preemptions_workload_total",
                        metrics_response
                    )
                )

                self.finished.append(
                    find_prometheus_metric_value(
                        f"request_success_total",
                        metrics_response
                    )
                )

                self.processed_tokens_prompt.append(
                    find_prometheus_metric_value(
                        f"prompt_tokens_total",
                        metrics_response
                    )
                )
                self.processed_tokens_generation.append(
                    find_prometheus_metric_value(
                        f"generation_tokens_total",
                        metrics_response
                    )
                )

                for index in range(len(self.available_loras) + 1):
                    self.running_by_adapter[index].append(
                        find_prometheus_metric_value(
                            f"running_by_adapter_{index}",
                            metrics_response
                        )
                    )
                    self.waiting_by_adapter[index].append(
                        find_prometheus_metric_value(
                            f"waiting_by_adapter_{index}",
                            metrics_response
                        )
                    )

                '''
                for user in self.users:
                    self.vtc_cost_by_user[user].append(
                        find_prometheus_metric_value(
                            f"vtc_cost_by_user_{user}",
                            metrics_response
                        )
                    )
                    self.vtc_waiting_by_user[user].append(
                        find_prometheus_metric_value(
                            f"vtc_waiting_by_user_{user}",
                            metrics_response
                        )
                    )
                    self.vtc_running_by_user[user].append(
                        find_prometheus_metric_value(
                            f"vtc_running_by_user_{user}",
                            metrics_response
                        )
                    )

                for lora in self.available_loras:
                    self.polling_waiting_by_queue[lora].append(
                        find_prometheus_metric_value(
                            f"polling_waiting_by_queue_{lora}",
                            metrics_response
                        )
                    )
                    self.polling_running_by_queue[lora].append(
                        find_prometheus_metric_value(
                            f"polling_running_by_queue_{lora}",
                            metrics_response
                        )
                    )
                '''

            except Exception as e:
                print(f'Error while running usage checker: {e}')
            finally:
                time.sleep(1)

    def __signal_term_handler(self, signal, frame):
        self.__save_metrics()
        sys.exit(0)

    def __save_metrics(self):
        np.save(
            os.path.join(self.output_path, f'time'),
            np.asarray(self.time)
        )
        np.save(
            os.path.join(self.output_path, f'gpu_cache_usage_perc'),
            np.asarray(self.gpu_cache_usage_perc)
        )
        np.save(
            os.path.join(self.output_path, f'num_running'),
            np.asarray(self.num_running)
        )
        np.save(
            os.path.join(self.output_path, f'num_waiting'),
            np.asarray(self.num_waiting)
        )
        np.save(
            os.path.join(self.output_path, f'num_preempted'),
            np.asarray(self.num_preempted)
        )
        np.save(
            os.path.join(self.output_path, f'num_preempted_workload'),
            np.asarray(self.num_preempted_workload)
        )
        np.save(
            os.path.join(self.output_path, f'finished'),
            np.asarray(self.finished)
        )
        np.save(
            os.path.join(self.output_path, f'processed_tokens_prompt'),
            np.asarray(self.processed_tokens_prompt)
        )
        np.save(
            os.path.join(self.output_path, f'processed_tokens_token'),
            np.asarray(self.processed_tokens_generation)
        )

        for index in range(len(self.available_loras) + 1):
            running_by_adapter = np.asarray(self.running_by_adapter[index])
            if np.count_nonzero(running_by_adapter) > 0:
                np.save(
                    os.path.join(self.output_path, f'running_by_adapter_{index}'),
                    running_by_adapter
                )
            waiting_by_adapter = np.asarray(self.waiting_by_adapter[index])
            if np.count_nonzero(waiting_by_adapter) > 0:
                np.save(
                    os.path.join(self.output_path, f'waiting_by_adapter_{index}'),
                    waiting_by_adapter
                )

        '''
        for lora in self.available_loras:
            np.save(
                os.path.join(self.output_path, f'polling_waiting_by_queue_{lora}'),
                np.asarray(self.polling_waiting_by_queue[lora])
            )
            np.save(
                os.path.join(self.output_path, f'polling_running_by_queue_{lora}'),
                np.asarray(self.polling_running_by_queue[lora])
            )

        for user in self.users:
            np.save(
                os.path.join(self.output_path, f'vtc_cost_by_user_{user}'),
                np.asarray(self.vtc_cost_by_user[user])
            )
            np.save(
                os.path.join(self.output_path, f'vtc_waiting_by_user_{user}'),
                np.asarray(self.vtc_waiting_by_user[user])
            )
            np.save(
                os.path.join(self.output_path, f'vtc_running_by_user_{user}'),
                np.asarray(self.vtc_running_by_user[user])
            )
        '''
