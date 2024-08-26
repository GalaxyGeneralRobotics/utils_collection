from datetime import datetime
from time import sleep
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Wait for GPU processes to finish")
parser.add_argument("--try_num", type=int, default=60, help="GPU IDs to monitor")
parser.add_argument("--gpu", nargs="+", type=int, help="GPU IDs to monitor")
args = parser.parse_args()

def get_gpu_process_count():
    try:
        # Run gpustat and capture the output
        result = subprocess.run(['gpustat'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if gpustat ran successfully
        if result.returncode != 0:
            print(f"Error running gpustat: {result.stderr}")
            return
        
        # Get the output
        output = result.stdout
        
        # Split the output into lines
        lines = output.splitlines()
        
        # Initialize process count
        process_count = []
        
        # Iterate over the lines to count GPU processes
        for line in lines:
            # Skip the header line
            if not line.startswith("["):
                continue
            if int(line[1]) not in args.gpu:
                continue
            
            percent = int(line.split("|")[1].split(',')[-1].split('%')[0].strip())
            # print(line[1], percent)
            if percent != 0:
                process_count.append((line[1], f'{percent}%'))
        
    
    except Exception as e:
        print(f"An error occurred: {e}")
        process_count = -1

    print("Current time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f", GPU process count: {process_count}", end="\r")
    return len(process_count)

if __name__ == "__main__":
    while True:
        while True:
            count = get_gpu_process_count()
            if count == 0:
                break
            sleep(60)

        empty = True
        for _ in range(args.try_num):
            count = get_gpu_process_count()
            if count != 0:
                empty = False
                break
            sleep(10)
        if empty:
            break
