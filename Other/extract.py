def extract_gpu_time(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    
    gpu_times = []
    for line in lines:
        if line.startswith("GPU Time:"):
            gpu_time = line.strip().split(" ")[-2] 
            gpu_times.append(gpu_time)
    
    return gpu_times

def main():
    log_file = 'nv_merge logs vs flims.txt'
    gpu_times = extract_gpu_time(log_file)
    for time in gpu_times:
        print(time)

if __name__ == "__main__":
    main()
