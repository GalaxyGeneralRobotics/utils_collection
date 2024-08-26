import os
import sys
import argparse
import subprocess
import signal

# Set the root path
ROOT_PATH = os.path.abspath(__file__)
for _ in range(3):
    ROOT_PATH = os.path.dirname(ROOT_PATH)
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)

# Argument parsing
parser = argparse.ArgumentParser(description="Run commands in parallel.")
parser.add_argument("--command", type=str, required=True, help="The command to run.")
parser.add_argument("--total", type=int, default=10, help="Number of parts to run in parallel.")
args = parser.parse_args()

# List to store subprocesses
processes = []

# Function to handle termination signals
def signal_handler(sig, frame):
    print("Terminating all processes...")
    for p in processes:
        p.kill()  # Use kill() to forcefully terminate the process
    sys.exit(0)

# Register the signal handler for SIGINT and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    # Run the command in parallel
    for i in range(args.total):
        command = f'{args.command} --part {i} --total {args.total}'
        p = subprocess.Popen(command, shell=True)
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.wait()

finally:
    # Ensure all processes are terminated if the script exits unexpectedly
    for p in processes:
        if p.poll() is None:  # Check if the process is still running
            p.kill()
