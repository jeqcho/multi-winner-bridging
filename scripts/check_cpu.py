#!/usr/bin/env python3
"""
Check CPU utilization and system resources.

Shows:
- CPU core count and architecture
- Current CPU usage breakdown
- Load average
- Top CPU-consuming processes
- Memory usage

Usage:
    python scripts/check_cpu.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "N/A"


def get_cpu_info():
    """Get CPU architecture information."""
    cpu_info = {}
    
    # Number of cores
    cpu_info['cores'] = run_command("nproc")
    
    # Detailed CPU info
    lscpu_output = run_command("lscpu")
    if lscpu_output != "N/A":
        for line in lscpu_output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key in ['CPU(s)', 'Thread(s) per core', 'Core(s) per socket', 'Socket(s)']:
                    cpu_info[key] = value
    
    return cpu_info


def get_cpu_usage():
    """Get current CPU usage from top."""
    output = run_command("top -bn1 | grep '%Cpu'")
    if output != "N/A":
        # Parse: %Cpu(s):  2.9 us,  0.2 sy,  0.0 ni, 96.2 id,  0.7 wa,  0.0 hi,  0.0 si,  0.0 st
        parts = output.split(',')
        usage = {}
        for part in parts:
            part = part.strip()
            if ':' in part:
                # First part has the label
                label_part = part.split(':')[1] if ':' in part else part
            if '%' in part:
                key_value = part.split('%')[0].strip().split()
                if len(key_value) == 2:
                    value, key = key_value
                    usage[key] = f"{value}%"
        return usage
    return {}


def get_load_average():
    """Get load average."""
    uptime_output = run_command("uptime")
    if uptime_output != "N/A":
        # Extract load average: "load average: 0.74, 0.28, 0.11"
        if "load average:" in uptime_output:
            load_part = uptime_output.split("load average:")[1].strip()
            loads = [x.strip() for x in load_part.split(',')]
            return {
                '1min': loads[0] if len(loads) > 0 else "N/A",
                '5min': loads[1] if len(loads) > 1 else "N/A",
                '15min': loads[2] if len(loads) > 2 else "N/A"
            }
    return {}


def get_top_processes(n=5):
    """Get top N CPU-consuming processes."""
    output = run_command(f"top -bn1 | head -20 | tail -n +8")
    if output != "N/A":
        processes = []
        lines = output.split('\n')[:n]
        for line in lines:
            parts = line.split()
            if len(parts) >= 12:
                pid = parts[0]
                user = parts[1]
                cpu = parts[8]
                mem = parts[9]
                cmd = ' '.join(parts[11:])
                processes.append({
                    'pid': pid,
                    'user': user,
                    'cpu': f"{cpu}%",
                    'mem': f"{mem}%",
                    'cmd': cmd[:60]  # Truncate long commands
                })
        return processes
    return []


def get_memory_info():
    """Get memory usage information."""
    output = run_command("top -bn1 | grep 'MiB Mem'")
    if output != "N/A":
        # Parse: MiB Mem :  63252.4 total,  60221.0 free,   1668.9 used,   2018.8 buff/cache
        parts = output.split(',')
        mem_info = {}
        for part in parts:
            part = part.strip()
            if 'total' in part:
                mem_info['total'] = part.split()[0] + " MiB"
            elif 'free' in part:
                mem_info['free'] = part.split()[0] + " MiB"
            elif 'used' in part:
                mem_info['used'] = part.split()[0] + " MiB"
            elif 'buff/cache' in part:
                mem_info['buff/cache'] = part.split()[0] + " MiB"
        return mem_info
    return {}


def main():
    print("=" * 70)
    print("CPU UTILIZATION REPORT")
    print("=" * 70)
    
    # CPU Architecture
    print("\n📊 CPU ARCHITECTURE:")
    print("-" * 70)
    cpu_info = get_cpu_info()
    if cpu_info:
        print(f"  Total CPU cores: {cpu_info.get('cores', 'N/A')}")
        if 'CPU(s)' in cpu_info:
            print(f"  CPU(s): {cpu_info['CPU(s)']}")
        if 'Thread(s) per core' in cpu_info:
            print(f"  Thread(s) per core: {cpu_info['Thread(s) per core']}")
        if 'Core(s) per socket' in cpu_info:
            print(f"  Core(s) per socket: {cpu_info['Core(s) per socket']}")
        if 'Socket(s)' in cpu_info:
            print(f"  Socket(s): {cpu_info['Socket(s)']}")
    
    # CPU Usage
    print("\n⚡ CPU USAGE:")
    print("-" * 70)
    cpu_usage = get_cpu_usage()
    if cpu_usage:
        print(f"  User space:   {cpu_usage.get('us', 'N/A'):>6}")
        print(f"  System:       {cpu_usage.get('sy', 'N/A'):>6}")
        print(f"  Idle:         {cpu_usage.get('id', 'N/A'):>6}")
        print(f"  I/O Wait:     {cpu_usage.get('wa', 'N/A'):>6}")
        print(f"  Nice:         {cpu_usage.get('ni', 'N/A'):>6}")
        print(f"  Hardware IRQ: {cpu_usage.get('hi', 'N/A'):>6}")
        print(f"  Software IRQ: {cpu_usage.get('si', 'N/A'):>6}")
        print(f"  Steal:        {cpu_usage.get('st', 'N/A'):>6}")
    else:
        # Fallback to simpler method
        cpu_percent = run_command("top -bn1 | grep '%Cpu'")
        if cpu_percent != "N/A":
            print(f"  {cpu_percent}")
    
    # Load Average
    print("\n📈 LOAD AVERAGE:")
    print("-" * 70)
    load = get_load_average()
    if load:
        print(f"  1 minute:  {load.get('1min', 'N/A'):>8}")
        print(f"  5 minutes: {load.get('5min', 'N/A'):>8}")
        print(f"  15 minutes: {load.get('15min', 'N/A'):>8}")
    else:
        uptime = run_command("uptime")
        if uptime != "N/A":
            print(f"  {uptime}")
    
    # Memory
    print("\n💾 MEMORY:")
    print("-" * 70)
    mem_info = get_memory_info()
    if mem_info:
        print(f"  Total:      {mem_info.get('total', 'N/A'):>12}")
        print(f"  Used:       {mem_info.get('used', 'N/A'):>12}")
        print(f"  Free:       {mem_info.get('free', 'N/A'):>12}")
        print(f"  Buff/Cache: {mem_info.get('buff/cache', 'N/A'):>12}")
    
    # Top Processes
    print("\n🔥 TOP CPU PROCESSES:")
    print("-" * 70)
    processes = get_top_processes(5)
    if processes:
        print(f"{'PID':<8} {'USER':<10} {'CPU%':<8} {'MEM%':<8} {'COMMAND'}")
        print("-" * 70)
        for proc in processes:
            print(f"{proc['pid']:<8} {proc['user']:<10} {proc['cpu']:<8} {proc['mem']:<8} {proc['cmd']}")
    else:
        # Fallback
        top_output = run_command("ps aux --sort=-%cpu | head -6")
        if top_output != "N/A":
            print(top_output)
    
    # Utilization Summary
    print("\n📊 UTILIZATION SUMMARY:")
    print("-" * 70)
    cores = int(cpu_info.get('cores', 0)) if cpu_info.get('cores', 'N/A') != 'N/A' else 0
    if cores > 0:
        idle_pct = float(cpu_usage.get('id', '0').rstrip('%')) if cpu_usage.get('id') else 0
        used_pct = 100 - idle_pct
        utilization = used_pct / 100
        
        print(f"  Total cores available: {cores}")
        print(f"  Current utilization: {used_pct:.1f}% ({utilization * cores:.1f} cores in use)")
        print(f"  Available capacity: {idle_pct:.1f}% ({(idle_pct / 100) * cores:.1f} cores idle)")
        
        if utilization < 0.1:
            print(f"\n  ⚠️  WARNING: Only using {utilization * 100:.1f}% of CPU capacity!")
            print(f"     Consider parallelizing your workload to use all {cores} cores.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

