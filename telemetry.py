import argparse
import csv
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict

import psutil
import pynvml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

@dataclass
class TelemetryData:
    # Shared data structure to hold the latest metrics
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    # GPU data stored as a list of dicts (one per GPU)
    gpu_metrics: List[Dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

def find_process_by_name(name):
    """Find the first process matching the given name."""
    for proc in psutil.process_iter(['pid', 'name']):
        if name.lower() in proc.info['name'].lower():
            return proc
    return None

def cpu_worker(process, interval, data_store):
    """Thread for collecting aggregated CPU and RAM telemetry for process and all children."""
    while True:
        try:
            # Get the main process + all recursive children
            all_procs = [process] + process.children(recursive=True)
            total_cpu = 0.0
            total_ram = 0.0
            
            for p in all_procs:
                try:
                    total_cpu += p.cpu_percent(interval=None)
                    total_ram += p.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            with data_store.lock:
                data_store.cpu_percent = total_cpu
                data_store.ram_mb = total_ram
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print("\n[!] Main process lost. Stopping CPU worker.")
            break
        time.sleep(interval)

def gpu_worker(target_pid, interval, data_store):
    """Thread for collecting GPU telemetry for the process tree."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    while True:
        try:
            # Find all PIDs in the process tree to check against the GPU
            parent = psutil.Process(target_pid)
            relevant_pids = {parent.pid} | {p.pid for p in parent.children(recursive=True)}
            
            current_gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                # Check if ANY PID in our process tree is using this GPU
                if any(p.pid in relevant_pids for p in procs):
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    
                    current_gpus.append({
                        'id': i, 'util': util.gpu, 'vram': mem.used / (1024 * 1024), 'power': power
                    })
            
            with data_store.lock:
                data_store.gpu_metrics = current_gpus
                
        except (pynvml.NVMLError, psutil.NoSuchProcess) as e:
            print(f"\n[!] GPU/Process Error: {e}")
            break
        time.sleep(interval)

def logger_worker(target_pid, interval, data_store, filename):
    """Thread for writing telemetry to a CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['timestamp', 'cpu_percent', 'ram_mb']
        # We don't know how many GPUs there are until runtime, but we'll add them
        writer.writerow(header)
        
        while True:
            with data_store.lock:
                timestamp = time.time()
                row = [timestamp, data_store.cpu_percent, data_store.ram_mb]
                # Append GPU data for all detected GPUs
                for gpu in data_store.gpu_metrics:
                    row.extend([f"gpu{gpu['id']}_util", gpu['util'], f"gpu{gpu['id']}_vram", gpu['vram'], f"gpu{gpu['id']}_power", gpu['power']])
                
                writer.writerow(row)
                f.flush()
            time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Process Telemetry Collector")
    parser.add_argument("process_name", help="Name of the process to monitor")
    parser.add_argument("-f", "--frequency", type=float, default=1.0, help="Sampling frequency in seconds (default: 1.0)")
    parser.add_argument("-o", "--output", default="telemetry_log.csv", help="Output CSV file (default: telemetry_log.csv)")
    args = parser.parse_args()

    process = find_process_by_name(args.process_name)
    if not process:
        print(f"Error: Process '{args.process_name}' not found.")
        sys.exit(1)

    print(f"Monitoring {process.name()} (PID: {process.pid})")
    
    data_store = TelemetryData()
    
    # Start Workers
    threads = []
    # CPU Thread
    t_cpu = threading.Thread(target=cpu_worker, args=(process, args.frequency, data_store), daemon=True)
    t_cpu.start()
    threads.append(t_cpu)
    
    # GPU Thread
    t_gpu = threading.Thread(target=gpu_worker, args=(process.pid, args.frequency, data_store), daemon=True)
    t_gpu.start()
    threads.append(t_gpu)
    
    # Logger Thread
    t_log = threading.Thread(target=logger_worker, args=(process.pid, args.frequency, data_store, args.output), daemon=True)
    t_log.start()
    threads.append(t_log)

    # --- Plotting Section (Must be in Main Thread) ---
    plt.style.use('ggplot')
    ##################################################################################
    ##  old plot initialization that resulted in cut-off labels
    ##################################################################################
    # fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    # plt.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.9)
    
    # # Data buffers for plotting
    # x_data = []
    # y_cpu = []
    # y_ram = []
    # y_vram = []
    # y_gpu_util = [] # Simplification: tracks the first GPU found
    
    # start_time = time.time()
    ##################################################################################

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9)
    
    # Initialize empty lines. We store them in a list to update them later.
    # This is MUCH more efficient than cla() and prevents labels from jumping/clipping.
    line_cpu, = axs[0].plot([], [], color='r')
    line_ram, = axs[1].plot([], [], color='b', label='System RAM')
    line_vram, = axs[1].plot([], [], color='g', label='GPU VRAM')
    line_gpu, = axs[2].plot([], [], color='b')

    # Set static labels once so they never move or get deleted
    axs[0].set_ylabel('CPU %')
    axs[0].set_title(f'Telemetry for {args.process_name}')
    axs[1].set_ylabel('Memory (MB)')
    axs[1].legend(loc='upper left')
    axs[2].set_ylabel('GPU Util %')
    axs[2].set_xlabel('Time (s)')
    plt.tight_layout(pad=4.0)

    # Data buffers for plotting
    x_data, y_cpu, y_ram, y_gpu_util, y_vram = [], [], [], [], []
    start_time = time.time()

    ##################################################################################
    ##  old update loop
    ##################################################################################
    # def update(frame):
    #     with data_store.lock:
    #         curr_cpu = data_store.cpu_percent
    #         curr_ram = data_store.ram_mb
    #         # Take the first GPU's utilization if any exist
    #         curr_gpu = data_store.gpu_metrics[1]['util'] if data_store.gpu_metrics else 0
    #         curr_vram = data_store.gpu_metrics[1]['vram'] if data_store.gpu_metrics else 0
    #         
    #     x_data.append(time.time() - start_time)
    #     y_cpu.append(curr_cpu)
    #     y_ram.append(curr_ram)
    #     y_vram.append(curr_vram)
    #     y_gpu_util.append(curr_gpu)
    #     
    #     # Keep only last 60 seconds of data
    #     if len(x_data) > (60 / args.frequency):
    #         x_data.pop(0)
    #         y_cpu.pop(0)
    #         y_ram.pop(0)
    #         y_gpu_util.pop(0)
    #         y_vram.pop(0)

    #     axs[0].cla()
    #     axs[0].plot(x_data, y_cpu, color='r')
    #     axs[0].set_ylabel('CPU %')
    #     axs[0].set_title(f'Telemetry for {args.process_name}')

    #     axs[1].cla()
    #     axs[1].plot(x_data, y_ram, color='b')
    #     axs[1].plot(x_data, y_vram, color='g')
    #     axs[1].set_ylabel('Memory (MB)')
    #     axs[1].legend(loc='upper left')

    #     axs[2].cla()
    #     axs[2].plot(x_data, y_gpu_util, color='b')
    #     axs[2].set_ylabel('GPU Util %')
    #     axs[2].set_xlabel('Time (s)')
    ##################################################################################

    def update(frame):
        with data_store.lock:
            curr_cpu = data_store.cpu_percent
            curr_ram = data_store.ram_mb
            curr_gpu = data_store.gpu_metrics[0]['util'] if data_store.gpu_metrics else 0
            curr_vram = data_store.gpu_metrics[0]['vram'] if data_store.gpu_metrics else 0
            
        x_data.append(time.time() - start_time)
        y_cpu.append(curr_cpu)
        y_ram.append(curr_ram)
        y_gpu_util.append(curr_gpu)
        y_vram.append(curr_vram)
        
        if len(x_data) > (60 / args.frequency):
            x_data.pop(0); y_cpu.pop(0); y_ram.pop(0); y_gpu_util.pop(0); y_vram.pop(0)

        # Update the data of the existing lines instead of clearing the whole plot
        line_cpu.set_data(x_data, y_cpu)
        line_ram.set_data(x_data, y_ram)
        line_vram.set_data(x_data, y_vram)
        line_gpu.set_data(x_data, y_gpu_util)

        # Dynamically adjust the X-axis window to slide with time
        for ax in axs:
            ax.set_xlim(x_data[0], x_data[-1])
            if ax == axs[0]: 
                ax.set_ylim(min(y_cpu)-1, max(y_cpu)+1)
            if ax == axs[1]: 
                # Ensure a minimum floor of 1.0 to prevent the line from disappearing at 0
                ax.set_ylim(0, max(1.0, max(max(y_ram), max(y_vram)) * 1.1))
            if ax == axs[2]: 
                ax.set_ylim(min(y_gpu_util)-1, max(y_gpu_util)+1)

        return line_cpu, line_ram, line_vram, line_gpu

    ani = FuncAnimation(fig, update, interval=int(args.frequency * 1000))
    plt.show()

if __name__ == "__main__":
    main()