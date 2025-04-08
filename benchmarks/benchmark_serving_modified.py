# SPDX-License-Identifier: Apache-2.0
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import asyncio
import gc
import json
import random
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, List, Tuple, Dict
import shlex
import requests
import subprocess
from subprocess import Popen
from concurrent_metrics_checker import ConcurrentMetricsChecker
from multi_server import MultiServer
from router_server import RouterServer
from server import Server

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_dataset import (BurstGPTDataset, HuggingFaceDataset,
                               RandomDataset, SampleRequest, ShareGPTDataset,
                               SonnetDataset, VisionArenaDataset)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    input_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    mean_ttfts_ms_by_lora: Dict[str, float]
    mean_ttfts_ms_by_user: Dict[str, float]
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


async def get_request(
    input_requests: list[SampleRequest],
    input_requests_loras: List[str] = None,
    input_requests_users: List[str] = None,
    request_rate: float = float("inf"),
    burstiness: float = 1.0,
) -> AsyncGenerator[tuple[SampleRequest, Optional[str], Optional[str]], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampleRequest.
        input_requests_loras:
            A list of LoRA adapter names to use for each request.
        input_requests_users:
            A list of user IDs to associate with each request.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    input_requests: Iterable[SampleRequest] = iter(input_requests)

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)

    for index, request in enumerate(input_requests):
        lora = input_requests_loras[index] if input_requests_loras else None
        user = input_requests_users[index] if input_requests_users else None
        yield request, lora, user

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
    input_requests_loras: List[str] = None,
    input_requests_users: List[str] = None,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    # Add tracking for LoRAs and users
    ttfts_by_lora: Dict[str, List[float]] = {}
    ttfts_by_user: Dict[str, List[float]] = {}
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens

            if output_len is None:
                # We use the tokenizer to count the number of output tokens
                # for some serving backends instead of looking at
                # len(outputs[i].itl) since multiple output tokens may be
                # bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(outputs[i].generated_text,
                              add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            # Track ttfts by LoRA and user if available
            if input_requests_loras:
                request_lora: str = input_requests_loras[i]
                if request_lora in ttfts_by_lora:
                    ttfts_by_lora[request_lora] += [outputs[i].ttft]
                else:
                    ttfts_by_lora[request_lora] = [outputs[i].ttft]
                    
            if input_requests_users:
                request_user: str = input_requests_users[i]
                if request_user in ttfts_by_user:
                    ttfts_by_user[request_user] += [outputs[i].ttft]
                else:
                    ttfts_by_user[request_user] = [outputs[i].ttft]
                    
            completed += 1
        else:
            actual_output_lens.append(0)
            # Add LoRA and user tracking for failed requests
            if input_requests_loras:
                request_lora: str = input_requests_loras[i]
                ttft_value = dur_s  # Use duration as TTFT for failed requests
                if request_lora in ttfts_by_lora:
                    ttfts_by_lora[request_lora] += [ttft_value]
                else:
                    ttfts_by_lora[request_lora] = [ttft_value]
                    
            if input_requests_users:
                request_user: str = input_requests_users[i]
                ttft_value = dur_s  # Use duration as TTFT for failed requests
                if request_user in ttfts_by_user:
                    ttfts_by_user[request_user] += [ttft_value]
                else:
                    ttfts_by_user[request_user] = [ttft_value]

    # Calculate mean TTFTs by LoRA and user
    mean_ttfts_by_lora: Dict[str, float] = {}
    for lora, values in ttfts_by_lora.items():
        mean_ttfts_by_lora[lora] = np.mean(values or 0) * 1000
        
    mean_ttfts_by_user: Dict[str, float] = {}
    for user, values in ttfts_by_user.items():
        mean_ttfts_by_user[user] = np.mean(values or 0) * 1000

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        mean_ttfts_ms_by_lora=mean_ttfts_by_lora,
        mean_ttfts_ms_by_user=mean_ttfts_by_user,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )

    return metrics, actual_output_lens

async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    lora_modules: Optional[Iterable[str]] = None,
    input_requests_loras: List[str] = None,
    input_requests_users: List[str] = None,
    infinite_behaviour: bool = False,
    lora_pre_loading: bool = False,
):
    """
    Run the benchmark with the given parameters.
    
    Args:
        backend: Backend to use for inference
        api_url: API URL for sending requests
        base_url: Base URL for the server
        model_id: Model ID to use
        model_name: Model name for display/reporting
        tokenizer: Tokenizer instance
        input_requests: List of requests to benchmark
        logprobs: Number of logprobs to return, if any
        request_rate: Requests per second rate
        burstiness: Burstiness factor for request generation
        disable_tqdm: Whether to disable progress bar
        profile: Whether to use profiling
        selected_percentile_metrics: Metrics to report percentiles for
        selected_percentiles: Percentiles to report
        ignore_eos: Whether to ignore EOS tokens
        goodput_config_dict: Configuration for goodput calculation
        max_concurrency: Maximum number of concurrent requests
        lora_modules: LoRA modules to use (random assignment)
        input_requests_loras: LoRA adapter names for each request (explicit assignment)
        input_requests_users: User IDs for each request
        infinite_behaviour: Whether to use infinite benchmark behavior 
        lora_pre_loading: Whether to pre-load a LoRA module
    """
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len, test_output_len, test_mm_content = \
        input_requests[0].prompt, input_requests[0].prompt_len, \
        input_requests[0].expected_output_len, \
            input_requests[0].multi_modal_data

    if backend != "openai-chat" and test_mm_content is not None:
        # multi-modal benchmark is only available on OpenAI Chat backend.
        raise ValueError(
            "Multi-modal content is only supported on 'openai-chat' backend.")
    assert test_mm_content is None or isinstance(test_mm_content, dict)
   
    # Handle LoRA and user information for test request
    test_model_id = model_id
    if lora_pre_loading and input_requests_loras and len(input_requests_loras) > 0:
        test_model_id = input_requests_loras[0]
    
    test_user = None
    if input_requests_users and len(input_requests_users) > 0:
        test_user = input_requests_users[0]

    test_input = RequestFuncInput(
        model=test_model_id,
        model_name=model_name,
        prompt=test_prompt,
        api_url=api_url,
        prompt_len=test_prompt_len,
        output_len=test_output_len,
        logprobs=logprobs,
        multi_modal_content=test_mm_content,
        ignore_eos=ignore_eos
    )

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    # Handle random LoRA module assignment if not explicitly provided
    if lora_modules and not input_requests_loras:
        input_requests_loras = [random.choice(list(lora_modules)) 
                               for _ in range(len(input_requests))]

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(model=model_id,
                                         model_name=model_name,
                                         prompt=test_prompt,
                                         api_url=base_url + "/start_profile",
                                         prompt_len=test_prompt_len,
                                         output_len=test_output_len,
                                         logprobs=logprobs,
                                         multi_modal_content=test_mm_content,
                                         ignore_eos=ignore_eos)
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else None)

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input,
                                      pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    time_to_send = len(input_requests) / request_rate if request_rate != float("inf") else 0
    async for request_data in get_request(input_requests, request_rate=request_rate, burstiness=burstiness,
                                    input_requests_loras=input_requests_loras,
                                    input_requests_users=input_requests_users,
                                    ):
        request, lora, user = request_data

        # Extract request details
        prompt = request.prompt
        prompt_len = request.prompt_len
        output_len = request.expected_output_len
        mm_content = request.multi_modal_data
        req_model_id, req_model_name = model_id, model_name
        if lora:
            req_model_id = lora
            if lora_pre_loading:
                req_model_name = lora

        request_func_input = RequestFuncInput(model=req_model_id,
                                              model_name=req_model_name,
                                              prompt=prompt,
                                              api_url=api_url,
                                              prompt_len=prompt_len,
                                              output_len=output_len,
                                              logprobs=logprobs,
                                              multi_modal_content=mm_content,
                                              ignore_eos=ignore_eos)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input,
                                     pbar=pbar)))
    # Handle different benchmark execution modes
    if infinite_behaviour:
        # For infinite mode: evaluate intermediate results after sending all requests
        remaining_time = time_to_send - (time.perf_counter() - benchmark_start_time)
        if remaining_time > 0:
            await asyncio.sleep(remaining_time)
            
        intermediate_benchmark_duration = time.perf_counter() - benchmark_start_time
        
        # Collect outputs from completed tasks
        outputs = []
        for task in tasks:
            if task.done():
                outputs.append(task.result())
            else:
                outputs.append(None)
        
        # Calculate metrics for completed requests
        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=intermediate_benchmark_duration,
            tokenizer=tokenizer,
            selected_percentile_metrics=selected_percentile_metrics,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
            input_requests_loras=input_requests_loras,
            input_requests_users=input_requests_users,
        )
        
        # Generate intermediate result
        result_intermediate = {
            "duration": intermediate_benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "request_goodput": metrics.request_goodput if goodput_config_dict else None,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "mean_ttft_ms": metrics.mean_ttft_ms,
            "mean_ttfts_ms_by_lora": metrics.mean_ttfts_ms_by_lora,
            "mean_ttfts_ms_by_user": metrics.mean_ttfts_ms_by_user,
            "median_ttft_ms": metrics.median_ttft_ms,
            "std_ttft_ms": metrics.std_ttft_ms,
            "mean_tpot_ms": metrics.mean_tpot_ms,
            "median_tpot_ms": metrics.median_tpot_ms,
            "std_tpot_ms": metrics.std_tpot_ms,
            "mean_itl_ms": metrics.mean_itl_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "std_itl_ms": metrics.std_itl_ms,
        }

        # Add percentile metrics if available
        for metric_name in selected_percentile_metrics:
            for p, value in getattr(metrics, f"percentiles_{metric_name}_ms", []):
                p_word = str(int(p)) if int(p) == p else str(p)
                result_intermediate[f"p{p_word}_{metric_name}_ms"] = value
        
        # Print intermediate results
        print("{s:{c}^{n}}".format(s=' Serving Benchmark Intermediate Result ', n=50, c='='))
        print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", intermediate_benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
        if goodput_config_dict:
            print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
        if hasattr(metrics, 'input_throughput'):
            print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", metrics.input_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
        
        # Cancel all tasks and prepare final result
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        result = {'infinite_behaviour': True}
        
        # Return both intermediate and final results
        return result_intermediate, result

    else:
        # For standard mode: wait for all tasks to complete
        outputs = await asyncio.gather(*tasks)
        
        if pbar:
            pbar.close()
        
        benchmark_duration = time.perf_counter() - benchmark_start_time
        
        # Calculate final metrics
        metrics, actual_output_lens = calculate_metrics(
            input_requests=input_requests,
            outputs=outputs,
            dur_s=benchmark_duration,
            tokenizer=tokenizer,
            selected_percentile_metrics=selected_percentile_metrics,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
            input_requests_loras=input_requests_loras,
            input_requests_users=input_requests_users,
        )
        
        # Print final results
        print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
        print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
        if goodput_config_dict:
            print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
        if hasattr(metrics, 'input_throughput'):
            print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", metrics.input_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
        print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))
        
        # Print detailed metrics for each selected metric type
        def print_metric_details(metric_name, metric_header):
            if metric_name in selected_percentile_metrics:
                print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
                print("{:<40} {:<10.2f}".format(f"Mean {metric_name.upper()} (ms):", 
                                              getattr(metrics, f"mean_{metric_name}_ms")))
                print("{:<40} {:<10.2f}".format(f"Median {metric_name.upper()} (ms):", 
                                              getattr(metrics, f"median_{metric_name}_ms")))
                for p, value in getattr(metrics, f"percentiles_{metric_name}_ms"):
                    p_word = str(int(p)) if int(p) == p else str(p)
                    print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name.upper()} (ms):", value))
        
        print_metric_details("ttft", "Time to First Token")
        print_metric_details("tpot", "Time per Output Token (excl. 1st token)")
        print_metric_details("itl", "Inter-token Latency")
        print_metric_details("e2el", "End-to-end Latency")

        print("=" * 50)
        
        # If profiling was enabled, stop it
        if profile:
            print("Stopping profiler...")
            profile_input = RequestFuncInput(
                model=model_id,
                prompt=test_prompt,
                api_url=base_url + "/stop_profile",
                prompt_len=test_prompt_len,
                output_len=test_output_len,
                logprobs=logprobs,
            )
            profile_output = await request_func(request_func_input=profile_input)
            if profile_output.success:
                print("Profiler stopped")
        
        # Prepare final result data
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "request_goodput": metrics.request_goodput if goodput_config_dict else None,
            "input_throughput": metrics.input_throughput,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "input_lens": [output.prompt_len for output in outputs],
            "output_lens": actual_output_lens,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            "generated_texts": [output.generated_text for output in outputs],
            "errors": [output.error for output in outputs],
        }

        # Add all the metrics to the result
        result["mean_ttft_ms"] = metrics.mean_ttft_ms
        result["median_ttft_ms"] = metrics.median_ttft_ms
        result["std_ttft_ms"] = metrics.std_ttft_ms
        result["mean_tpot_ms"] = metrics.mean_tpot_ms
        result["median_tpot_ms"] = metrics.median_tpot_ms
        result["std_tpot_ms"] = metrics.std_tpot_ms
        result["mean_itl_ms"] = metrics.mean_itl_ms
        result["median_itl_ms"] = metrics.median_itl_ms
        result["std_itl_ms"] = metrics.std_itl_ms
        result["mean_e2el_ms"] = metrics.mean_e2el_ms
        result["median_e2el_ms"] = metrics.median_e2el_ms
        result["std_e2el_ms"] = metrics.std_e2el_ms
        
        # Add LoRA and user-specific metrics if available
        if hasattr(metrics, 'mean_ttfts_ms_by_lora'):
            result["mean_ttfts_ms_by_lora"] = metrics.mean_ttfts_ms_by_lora
        if hasattr(metrics, 'mean_ttfts_ms_by_user'):
            result["mean_ttfts_ms_by_user"] = metrics.mean_ttfts_ms_by_user
        
        # Add percentile metrics
        for metric_name in selected_percentile_metrics:
            for p, value in getattr(metrics, f"percentiles_{metric_name}_ms", []):
                p_word = str(int(p)) if int(p) == p else str(p)
                result[f"p{p_word}_{metric_name}_ms"] = value
        
        # For compatibility with the infinite mode, return both intermediate and final results
        # (they're the same in standard mode)
        return result, result

def check_goodput_args(args):
    # Check and parse goodput arguments
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. ")
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative.")
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            "Specify service level objectives for goodput as \"KEY:VALUE\" "
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds.") from err
    return goodput_config_dict


def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, Any],
                                     file_name: str) -> None:
    metrics = [
        "median_ttft_ms", "mean_ttft_ms", "std_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "std_tpot_ms", "p99_tpot_ms",
        "median_itl_ms", "mean_itl_ms", "std_itl_ms", "p99_itl_ms"
    ]
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["ttfts", "itls", "generated_texts", "errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]]
                 for k in metrics},
        extra_info={
            k: results[k]
            for k in results if k not in metrics and k not in ignored_metrics
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)

def add_duplicated_loras_to_server_args(server_args, lora_path, n_loras, base_model):
    """
    Modify server arguments to include multiple copies of the same LoRA adapter
    with different names.
    
    Args:
        server_args: Original server arguments string
        lora_path: Path to the LoRA adapter file to duplicate
        n_loras: Number of times to duplicate the LoRA adapter
        
    Returns:
        Modified server arguments string with duplicated LoRAs
    """ 
    if "--enable-lora" not in server_args:
        server_args += " --enable-lora"\
        
    lora_names = []
    lora_entries = []

    for i in range(n_loras):
        lora_name = f"duplicated_lora_{i}"
        lora_names.append(lora_name)
        lora_entries.append(f"{lora_name}={lora_path}")

    server_args += " --lora-modules " + " ".join(lora_entries)
    print(server_args)
    return server_args,lora_names

def modify_server_args_with_lora_options(args):
    """
    Update server arguments based on lora-path and n-loras options.
    Check if server_args already contains LoRA modules and raise an error if it contains lora-path and n-loras too.
    
    Args:
        args: The command line arguments parsed by argparse
        
    Returns:
        Updated args with modified server_args if applicable
        
    Raises:
        ValueError: If server_args already contains LoRA modules
    """
    if hasattr(args, 'n_loras') and args.n_loras > 0 and "--lora-modules" in args.server_args and args.lora_modules:
        raise ValueError("Server arguments already contain LoRA modules. Cannot add duplicated LoRAs.")
    elif args.lora_path and args.n_loras > 0:
        print(f"Adding {args.n_loras} copies of LoRA adapter from {args.lora_path}")
        try:
            args.server_args, args.lora_modules = add_duplicated_loras_to_server_args(
                args.server_args,
                args.lora_path,
                args.n_loras,
                args.model
            )                
            print(f"Modified server args: {args.server_args} \n")
            print(f"Modified LoRA modules: {args.lora_modules} \n")
        except ValueError as e:
            print(f"ERROR: {e}")
            print("Please remove existing LoRA modules from server_args when using --lora-path and --n-loras.")
            return
    return args


def main(args: argparse.Namespace):
    """
    Main function to execute the benchmark based on command line arguments.
    
    This function handles:
    1. Setting up server if requested
    2. Configuring LoRA and user assignments
    3. Running the benchmark
    4. Saving results
    """
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
        metrics_api_url = f"{args.base_url}/metrics/"
        models_api_url = f"{args.base_url}/v1/models/"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"
        metrics_api_url = f"http://{args.host}:{args.port}/metrics/"
        models_api_url = f"http://{args.host}:{args.port}/v1/models/"

    # Helper functions for LoRA and user management
    def __get_available_loras(models_url, main_model_id):
        """Get available LoRA adapters from the server."""
        try:
            response = requests.get(models_url).json()
            available_loras = [model["id"] for model in response["data"]]
            if main_model_id not in available_loras:
                raise ValueError("Benchmarking with a model that is not available in the server")
            available_loras.remove(main_model_id)
            if len(available_loras) == 0:
                raise ValueError("No available LoRAs in the server")
            return available_loras
        except Exception as e:
            print(f"Error getting available LoRAs: {e}")
            return []
        
    def __assign_users_loras(input_requests, user_lora_request_relation, available_loras):
        """Assign users and LoRAs to requests based on the specified relation pattern."""
        input_requests_users = []
        input_requests_loras = []

        if user_lora_request_relation is None or user_lora_request_relation in ['default', 'balance']:
            # Balanced distribution of users and LoRAs
            index = 0
            while len(input_requests_users) < len(input_requests):
                input_requests_users.append(str(index))
                input_requests_loras.append(available_loras[index])
                index += 1
                if index >= len(available_loras):
                    index = 0
            aux_shuffled_list = list(zip(input_requests_users, input_requests_loras))
            random.shuffle(aux_shuffled_list)
            input_requests_users, input_requests_loras = zip(*aux_shuffled_list)
            input_requests_users = list(input_requests_users)
            input_requests_loras = list(input_requests_loras)
        elif user_lora_request_relation == 'imbalance':
            # Imbalanced distribution: some users make more requests
            index = 0
            while len(input_requests_users) < len(input_requests):
                input_requests_users.append(str(index))
                input_requests_loras.append(available_loras[index])
                if index % 2 == 0 and len(input_requests_users) < len(input_requests):
                    input_requests_users.append(str(index))
                    input_requests_loras.append(available_loras[index])
                index += 1
                if index >= len(available_loras):
                    index = 0
            aux_shuffled_list = list(zip(input_requests_users, input_requests_loras))
            random.shuffle(aux_shuffled_list)
            input_requests_users, input_requests_loras = zip(*aux_shuffled_list)
            input_requests_users = list(input_requests_users)
            input_requests_loras = list(input_requests_loras)
        else:
            raise ValueError(f"User assignation {user_lora_request_relation} not implemented")

        values, counts = np.unique(input_requests_users, return_counts=True)
        print(f"Requests users. Values: {values}. Counts: {counts}")
        values, counts = np.unique(input_requests_loras, return_counts=True)
        print(f"Requests loras. Values: {values}. Counts: {counts}")
        return input_requests_users, input_requests_loras

        
    tokenizer = get_tokenizer(tokenizer_id,
                              tokenizer_mode=tokenizer_mode,
                              trust_remote_code=args.trust_remote_code)

    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required.")

    if args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(num_requests=args.num_prompts,
                                            input_len=args.sonnet_input_len,
                                            output_len=args.sonnet_output_len,
                                            prefix_len=args.sonnet_prefix_len,
                                            tokenizer=tokenizer,
                                            return_prompt_formatted=False)
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset.")
            input_requests = dataset.sample(num_requests=args.num_prompts,
                                            input_len=args.sonnet_input_len,
                                            output_len=args.sonnet_output_len,
                                            prefix_len=args.sonnet_prefix_len,
                                            tokenizer=tokenizer,
                                            return_prompt_formatted=True)

    elif args.dataset_name == "hf":
        # Choose between VisionArenaDataset
        # and HuggingFaceDataset based on provided parameters.
        dataset_class = (VisionArenaDataset if args.dataset_path
                         == VisionArenaDataset.VISION_ARENA_DATASET_PATH
                         and args.hf_subset is None else HuggingFaceDataset)
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            random_seed=args.seed,
            output_len=args.hf_output_len,
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "sharegpt":
            lambda: ShareGPTDataset(random_seed=args.seed,
                                    dataset_path=args.dataset_path).sample(
                                        tokenizer=tokenizer,
                                        num_requests=args.num_prompts,
                                        output_len=args.sharegpt_output_len,
                                    ),
            "burstgpt":
            lambda: BurstGPTDataset(random_seed=args.seed,
                                    dataset_path=args.dataset_path).
            sample(tokenizer=tokenizer, num_requests=args.num_prompts),
            "random":
            lambda: RandomDataset(dataset_path=args.dataset_path).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
            )
        }

        try:
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err
    goodput_config_dict = check_goodput_args(args)

    # Avoid GC processing "static" data - reduce pause times.
    gc.collect()
    gc.freeze()

    # Initialize server, metrics collector and benchmark variables
    server = None
    open_server_process = None
    concurrent_metrics_checker = None

    try:
        # Launch server if requested
        if hasattr(args, 'launch_server') and args.launch_server:
            if args.num_servers > 1:
                server_ports = [args.port + i for i in range(args.num_servers)]
                # Use MultiServer for multiple servers
                print(f"Starting {args.num_servers} vLLM servers...")
                multi_server = MultiServer(args.server_args, args.result_dir, args.num_servers, args.port)
                open_server_processes = multi_server.run()

                # Get server URLs
                base_urls = multi_server.get_server_urls()
                api_urls = [f"{url}{args.endpoint}" for url in base_urls]
                metrics_api_urls = [f"{url}/metrics/" for url in base_urls]

                # Health check for all servers
                max_wait_for_server_seconds = 300
                init_time = time.time()
                
                # Track which servers have successfully started
                server_started_status = [False] * args.num_servers
                
                # Wait for all servers to start or timeout
                while not all(server_started_status) and time.time() - init_time < max_wait_for_server_seconds:
                    for i, metrics_url in enumerate(metrics_api_urls):
                        if not server_started_status[i]:
                            try:
                                response = requests.get(metrics_url, timeout=5)
                                if response.status_code == 200:
                                    server_started_status[i] = True
                                    print(f"Server {i} on port {args.port + i} started successfully")
                            except Exception:
                                # Continue if the server isn't ready yet
                                pass
                            
                    # If not all servers are up, wait before checking again
                    if not all(server_started_status):
                        time.sleep(5)

                # Check if all servers started successfully
                if not all(server_started_status):
                    # Identify which servers failed to start
                    failed_servers = [i for i, status in enumerate(server_started_status) if not status]
                    multi_server.terminate(open_server_processes)
                    raise Exception(f"Servers {failed_servers} did not start on time")

                print(f"All {args.num_servers} servers started successfully")
                
                # When using multiple servers, always use the router
                # Extract LoRA adapters info from server args
                available_loras = []
                if "--enable-lora" in args.server_args and "--lora-modules" in args.server_args:
                    arg_parts = args.server_args.split()
                    lora_modules_index = arg_parts.index("--lora-modules")

                    i = lora_modules_index + 1
                    while i < len(arg_parts) and not arg_parts[i].startswith("--"):
                        lora_info = arg_parts[i]
                        # Parse the format 'name=path'
                        if "=" in lora_info:
                            name, _ = lora_info.split("=", 1)
                            available_loras.append(name)
                        i += 1
                print(f"Extracted LoRA adapters for router: {available_loras}")

                # Start the router
                print("Starting vLLM router...")
                router_server = RouterServer(
                    output_path=args.result_dir,
                    router_port=args.router_port,
                    server_ports=server_ports,
                    base_model=model_id,
                    lora_adapters=available_loras
                )
                router_process = router_server.run()

                # Update API URL to point to the router
                api_url = f"http://{args.host}:{args.router_port}{args.endpoint}"
                base_url = f"http://{args.host}:{args.router_port}"
                models_api_url = f"http://{args.host}:{args.router_port}/v1/models/"

                # Wait for router to start
                max_wait_for_router_seconds = 60
                init_time = time.time()
                router_started = False

                while not router_started and time.time() - init_time < max_wait_for_router_seconds:
                    try:
                        response = requests.get(models_api_url, timeout=5)
                        if response.status_code == 200:
                            router_started = True
                            print("Router started successfully")
                        else:
                            time.sleep(2)
                    except Exception:
                        time.sleep(2)

                if not router_started:
                    raise Exception("Router failed to start within the timeout period")
            else:
                # Use original Server class for single server
                server = Server(args.server_args, args.result_dir)
                open_server_process = server.run()
                max_wait_for_server_seconds = 300
                init_time = time.time()
                server_started = False
                while not server_started and time.time() - init_time < max_wait_for_server_seconds:
                    try:
                        if requests.get(metrics_api_url).status_code == 200:
                            server_started = True
                        else:
                            time.sleep(5)
                    except Exception as e:
                        time.sleep(5)
                if not server_started:
                    raise Exception("Server did not start on time")
                print("Server started")

        # Set up request rate
        request_rate = args.request_rate
        
        # Handle LoRA/user assignment
        input_requests_users = None
        input_requests_loras = None
        all_available_loras = []
        
        if hasattr(args, 'disable_loras_users') and args.disable_loras_users or \
           hasattr(args, 'restrict_loras') and args.restrict_loras is not None and args.restrict_loras == 0:
            # Use default model without LoRAs
            input_requests_users = ["0"] * len(input_requests) 
            input_requests_loras = [model_id] * len(input_requests)
        else:
            # Get available LoRAs from server if models API is available
            try:
                all_available_loras = __get_available_loras(models_api_url, model_id)
                
                if hasattr(args, 'restrict_loras') and args.restrict_loras is not None:
                    if len(all_available_loras) < args.restrict_loras:
                        raise ValueError('Less available LoRAs than the ones that need to be restricted')
                    available_loras = all_available_loras[:args.restrict_loras]
                else:
                    available_loras = all_available_loras
                    
                input_requests_users, input_requests_loras = __assign_users_loras(
                    input_requests,
                    args.user_lora_request_relation if hasattr(args, 'user_lora_request_relation') else None,
                    available_loras
                )
                
                if hasattr(args, 'request_rate_by_lora') and args.request_rate_by_lora is not None:
                    request_rate = args.request_rate_by_lora * len(available_loras)
            except Exception as e:
                print(f"Warning: Failed to get LoRAs from server. Using default model. Error: {e}")
                input_requests_users = ["0"] * len(input_requests)
                input_requests_loras = [model_id] * len(input_requests)

        # Set up metrics checker if enabled
        if not args.disable_log_stats:
            if args.num_servers > 1:
                metrics_checkers = []
                try:
                    for i, server_metrics_url in enumerate(metrics_api_urls):
                        server_metrics_checker = ConcurrentMetricsChecker(
                            os.path.join(args.result_dir, f'server_{i}'),
                            server_metrics_url,
                            list(set(input_requests_users)) if input_requests_users else [],
                            all_available_loras
                        )
                        server_metrics_checker.start()
                        metrics_checkers.append(server_metrics_checker)
                        
                except Exception as e:
                    print(f"Warning: Failed to start server metrics checkers. Error: {e}")
            else:    
                try:
                    concurrent_metrics_checker = ConcurrentMetricsChecker(
                        args.result_dir,
                        metrics_api_url,
                        list(set(input_requests_users)) if input_requests_users else [],
                        all_available_loras
                    )
                    concurrent_metrics_checker.start()
                except Exception as e:
                    print(f"Warning: Failed to start metrics checker. Error: {e}")

        # Run the benchmark
        benchmark_result = asyncio.run(
            benchmark(
                backend=backend,
                api_url=api_url,
                base_url=base_url,
                model_id=model_id,
                model_name=model_name,
                tokenizer=tokenizer,
                input_requests=input_requests,
                logprobs=args.logprobs,
                request_rate=request_rate,
                burstiness=args.burstiness,
                disable_tqdm=args.disable_tqdm,
                profile=args.profile,
                selected_percentile_metrics=args.percentile_metrics.split(","),
                selected_percentiles=[
                    float(p) for p in args.metric_percentiles.split(",")
                ],
                ignore_eos=args.ignore_eos,
                goodput_config_dict=goodput_config_dict,
                max_concurrency=args.max_concurrency,
                lora_modules=args.lora_modules,
                infinite_behaviour=args.infinite_behaviour if hasattr(args, 'infinite_behaviour') else False,
                input_requests_loras=input_requests_loras,
                input_requests_users=input_requests_users,
                lora_pre_loading=args.lora_pre_loading if hasattr(args, 'lora_pre_loading') else False,
            ))

        # Save benchmark results
        if args.save_result:
            result_json = {}

            # Setup
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_json["date"] = current_dt
            result_json["backend"] = backend
            result_json["model_id"] = model_id
            result_json["tokenizer_id"] = tokenizer_id
            result_json["num_prompts"] = args.num_prompts

            # Metadata
            if args.metadata:
                for item in args.metadata:
                    if "=" in item:
                        kvstring = item.split("=")
                        result_json[kvstring[0].strip()] = kvstring[1].strip()
                    else:
                        raise ValueError(
                            "Invalid metadata format. Please use KEY=VALUE format."
                        )

            # Traffic
            result_json["request_rate"] = (args.request_rate if args.request_rate < float("inf") else "inf")
            if hasattr(args, 'burstiness'):
                result_json["burstiness"] = args.burstiness
            result_json["max_concurrency"] = args.max_concurrency

            # Get intermediate and final results from benchmark
            result_json_intermediate = {**result_json, **benchmark_result[0]}
            result_json_final = {**result_json, **benchmark_result[1]}

            # Save to file
            base_model_id = model_id.split("/")[-1]
            max_concurrency_str = (f"-concurrency{args.max_concurrency}"
                                 if args.max_concurrency is not None else "")
            
            # Save final result
            file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"
            if args.result_filename:
                file_name = args.result_filename
            if args.result_dir:
                file_name = os.path.join(args.result_dir, file_name)
            with open(file_name, "w", encoding='utf-8') as outfile:
                json.dump(result_json_final, outfile)
                
            # For infinite_behaviour or when intermediate results are different, save them too
            if hasattr(args, 'infinite_behaviour') and args.infinite_behaviour or benchmark_result[0] != benchmark_result[1]:
                intermediate_file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}_intermediate.json"
                if args.result_dir:
                    intermediate_file_name = os.path.join(args.result_dir, intermediate_file_name)
                with open(intermediate_file_name, "w", encoding='utf-8') as outfile:
                    json.dump(result_json_intermediate, outfile)
            
            # Save in PyTorch benchmark format if available
            if "save_to_pytorch_benchmark_format" in globals():
                save_to_pytorch_benchmark_format(args, result_json_final, file_name)

    finally:
        # Clean up resources
        try:
            if concurrent_metrics_checker is not None:
                concurrent_metrics_checker.shutdown()
                concurrent_metrics_checker.join(timeout=30)  # Wait up to 30 seconds
                if concurrent_metrics_checker.is_alive():
                    # Process didn't exit in time, force termination
                    concurrent_metrics_checker.terminate()
                print('Concurrent checker terminated')

            if 'metrics_checkers' in locals() and metrics_checkers:
                for checker in metrics_checkers:
                    checker.shutdown()
                    checker.join(timeout=30)
                    if checker.is_alive():
                        # Process didn't exit in time, force termination
                        checker.terminate()
                
        except Exception as e:
            print(f"Error while terminating metrics checker: {e}")
            
        try:
            # Router cleanup
            if args.num_servers > 1 and 'router_server' in locals():
                router_server.terminate()
                print('Router terminated')

            if args.num_servers > 1 and 'multi_server' in locals():
                multi_server.terminate(open_server_processes)
                print('All vLLM servers terminated')
            elif server and open_server_process:
                server.terminate(open_server_process)
                print('vLLM Server terminated')
        except Exception as e:
            print(f"Error while terminating servers: {e}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    #LoRA variables to load the same adapter as n different adapters.
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to where the LoRA adapter that will be duplicated --n-loras times is stored.",
    )
    parser.add_argument(
        "--n-loras",
        type=int,
        default=0,
        help="Number of times the LoRA adapter from --lora-path will be loaded, all as different adapters.",
    )
    # Router variables, used when using --num-servers > 1.
    parser.add_argument(
        "--router-port",
        type=int,
        default=8080,
        help="Port for the vllm_router service",
    )
    # MultiServer argument.
    parser.add_argument(
    "--num-servers",
    type=int,
    default=1,
    help="Number of vLLM server instances to launch (only used with --launch-server)",
    )
    # Add new server-related arguments
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch server in addition to benchmark",
    )
    parser.add_argument(
        "--server-args",
        type=str,
        default="",
        help="Args to send to the server when launching. Only useful when passing --launch-server as well",
    )
    
    # Add new LoRA and user-related arguments
    parser.add_argument(
        '--disable-log-stats',
        action='store_true',
        help='Disable logging statistics'
    )
    parser.add_argument(
        '--disable-loras-users',
        action='store_true',
        help='Only send requests without LoRA adapters'
    )
    parser.add_argument(
        "--user-lora-request-relation",
        type=str,
        default=None,
        help="Relation of lora<->request<->user. Options: default, balance, imbalance",
    )
    parser.add_argument(
        "--restrict-loras",
        type=int,
        default=None,
        help="Limit the number of available LoRAs to use",
    )
    parser.add_argument(
        '--lora-pre-loading',
        action='store_true',
        default=False,
        help='Pre-load one LoRA in the test previous to the real benchmark'
    )
    parser.add_argument(
        '--infinite-behaviour',
        action='store_true',
        default=False,
        help='Finish benchmark once all requests have been sent'
    )
    parser.add_argument(
        "--request-rate-by-lora",
        type=float,
        default=None,
        help="Number of requests per second by LoRA. Overall rate will be this value multiplied by the number of LoRAs."
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "burstgpt", "sonnet", "random", "hf"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the sharegpt/sonnet dataset. "
                        "Or the huggingface dataset ID if using HF dataset.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving the results, whether to include per request "
        "information such as response, error, ttfs, tpots, etc.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Specify service level objectives for goodput as \"KEY:VALUE\" "
        "pairs, where the key is a metric name, and the value is in "
        "milliseconds. Multiple \"KEY:VALUE\" pairs can be provided, "
        "separated by spaces. Allowed request level metric names are "
        "\"ttft\", \"tpot\", \"e2el\". For more context on the definition of "
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve")

    # group for dataset specific arguments
    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.")

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random sampling.",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens before random "
        " context. The length range of context in a random "
        " request is [random-prefix-len, "
        " random-prefix-len + random-prefix-len * random-range-ratio).")

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral', 'custom'],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        'always use the slow tokenizer. \n* '
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer.')

    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. "
                        "If not specified, the model name will be the "
                        "same as the ``--model`` argument. ")

    parser.add_argument("--lora-modules",
                        nargs='+',
                        default=None,
                        help="A subset of LoRA module names passed in when "
                        "launching the server. For each request, the "
                        "script chooses a LoRA module at random.")

    args = parser.parse_args()
    args=modify_server_args_with_lora_options(args)
    main(args)
