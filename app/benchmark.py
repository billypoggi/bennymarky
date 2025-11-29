import time
import psycopg2
from psycopg2 import sql
import math
import random
import platform
import subprocess 
import os 
import threading 
import sys 
import importlib.util

BENCHMARK_VERSION = "1.9"

# 1. Load Config from Environment Variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "postgres")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")

def get_connection():
    """Attempt to connect to the database with retries."""
    retries = 10
    while retries > 0:
        try:
            print(f"Connecting to {DB_NAME} at {DB_HOST}:{DB_PORT}...")
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASS
            )
            print("Successfully connected!")
            return conn
        except psycopg2.OperationalError as e:
            print(f"Database not ready yet... ({e})")
            retries -= 1
            time.sleep(2)  # Wait 2 seconds before retrying
    return None

def run_benchmark():
    conn = get_connection()
    if not conn:
        print("Could not connect to database. Exiting.")
        return

    conn.autocommit = True
    cursor = conn.cursor()

    # 2. Create a fresh table for testing
    print("Creating benchmark table...")
    cursor.execute("""
        DROP TABLE IF EXISTS cpu_benchmark;
        CREATE TABLE cpu_benchmark (
            id SERIAL PRIMARY KEY,
            data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. The Write Test (Insert 10,000 rows)
    print("Starting Write Benchmark (10,000 rows)...")
    start_time = time.time()
    
    # We use a loop here to simulate application load, rather than a single bulk insert
    for i in range(10000):
        cursor.execute("INSERT INTO cpu_benchmark (data) VALUES (%s)", (f'benchmark_data_{i}',))
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Benchmark Complete: {duration:.4f} seconds")
    print(f"Rows per second: {10000 / duration:.2f}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Small buffer to let Docker networking initialize
    time.sleep(1)
    run_benchmark()

# --- Robust Dependency Management ---
def check_and_install_dependencies():
    """
    Checks for required packages. If missing, installs them and RESTARTS the script.
    """
    # Map import name -> pip package name
    required_libraries = {
        'numpy': 'numpy',
        'torch': 'torch',
        'psycopg2': 'psycopg2-binary'
    }
    
    missing_packages = []

    # Check what is missing
    for import_name, install_name in required_libraries.items():
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(install_name)

    if not missing_packages:
        return True # All good

    print("-" * 50)
    print(f"[!] Missing critical libraries: {', '.join(missing_packages)}")
    print(f"[!] Downloading dependencies. PyTorch is large, please wait...")
    print("-" * 50)

    # Install loop
    success = True
    for package in missing_packages:
        print(f"[*] Installing {package}...")
        try:
            # Try standard install with output visible
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"[!] Standard install failed. Trying --user install for {package}...")
            try:
                # Fallback to user install
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])
            except subprocess.CalledProcessError as e:
                print(f"[x] Critical Error: Failed to install {package}. Details: {e}")
                success = False
                break
    
    if success:
        print("-" * 50)
        print("[+] Dependencies installed successfully.")
        print("[*] Restarting benchmark to load new libraries...")
        print("-" * 50)
        time.sleep(1)
        # Restart the script to ensure fresh import environment
        os.execv(sys.executable, [sys.executable] + sys.argv)
    else:
        print("[!] Automatic installation failed. Advanced features may be skipped.")
        return False

# Perform the check before anything else
if check_and_install_dependencies():
    try:
        import torch
        import numpy as np
        import psycopg2
        TORCH_AVAILABLE = True
        DB_AVAILABLE = True
    except ImportError:
        print("[!] Libraries installed but import failed. Try running the script again manually.")
        TORCH_AVAILABLE = False
        DB_AVAILABLE = False
else:
    TORCH_AVAILABLE = False
    DB_AVAILABLE = False


# --- Configuration ---
PRIME_TARGET_NUMBER = 1000000  
MATRIX_SIZE = 300              
MATRIX_ITERATIONS = 5          

# GPU Configuration 
GPU_MATRIX_SIZE = 4096
GPU_ITERATIONS = 20

def get_cpu_info():
    """
    Retrieves CPU information with robust fallbacks for Docker/Linux environments.
    """
    processor_name = None
    architecture = platform.machine() or "Unknown Architecture"
    system_os = platform.system()

    # 1. macOS (Darwin) specific detection (Host level)
    if system_os == "Darwin":
        try:
            command = ['sysctl', '-n', 'machdep.cpu.brand_string']
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                processor_name = result.stdout.strip()
        except Exception:
            pass
    
    # 2. Linux specific detection (Docker/VMs)
    elif system_os == "Linux":
        # Attempt A: Try 'lscpu' command (often best formatted)
        try:
            command = ['lscpu']
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                lscpu_output = result.stdout.splitlines()
                vendor_id = None
                model_name = None

                for line in lscpu_output:
                    if "Model name:" in line:
                        name = line.split(":", 1)[1].strip()
                        if name and name != "-" and "not implemented" not in name.lower():
                            model_name = name
                    if "Vendor ID:" in line:
                        vendor_id = line.split(":", 1)[1].strip()
                
                # Logic to construct the best possible name
                if model_name:
                    processor_name = model_name
                elif vendor_id == "Apple":
                    processor_name = "Apple Silicon (M-Series) [Specific Model Hidden by Docker VM]"
                elif vendor_id:
                     processor_name = f"{vendor_id} {architecture} (Model masked by VM)"

        except Exception:
            pass

        # Attempt B: /proc/cpuinfo (if lscpu failed or gave generic info)
        if not processor_name or "masked" in processor_name.lower():
            try:
                cpu_info = {}
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if ":" in line:
                            parts = line.split(":", 1)
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key not in cpu_info:
                                cpu_info[key] = value
                
                possible_keys = ["model name", "Hardware", "Processor", "model"]
                for key in possible_keys:
                    if key in cpu_info and len(cpu_info[key]) > 0:
                        new_name = cpu_info[key]
                        if "Apple" in processor_name and "ARM" in new_name:
                            continue 
                        processor_name = new_name
                        break
            except Exception:
                pass

    # 3. Final Fallback
    if not processor_name or processor_name.strip() == "":
        if architecture == "aarch64" and system_os == "Linux":
            processor_name = "ARM64 Processor (Host Model Masked by Container)"
        else:
            processor_name = platform.processor() or "Unknown/Generic Processor"
        
    return processor_name, architecture

def get_gpu_info():
    """
    Detects GPU using PyTorch if available, falling back to system commands.
    """
    gpu_name = "Not Detected"
    
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            gpu_name = "Apple Silicon GPU (MPS)"
        else:
            gpu_name = "No GPU acceleration found (PyTorch installed but running on CPU)"
    else:
        # Fallback to nvidia-smi if torch isn't there
        try:
            command = ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
            else:
                gpu_name = "Not Detected (Install 'torch' for accurate detection)"
        except Exception:
            pass
        
    return gpu_name

def calculate_primes(n):
    """Benchmarks integer performance."""
    start_time = time.time()
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for p in range(2, int(math.sqrt(n)) + 1):
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
                
    prime_count = sum(is_prime) 
    end_time = time.time()
    return end_time - start_time

def matrix_benchmark(size, iterations):
    """Benchmarks floating-point performance."""
    A = [[random.uniform(0.1, 10.0) for _ in range(size)] for _ in range(size)]
    B = [[random.uniform(0.1, 10.0) for _ in range(size)] for _ in range(size)]
    start_time = time.time()
    
    # Simple CPU Matrix Multiply
    for _ in range(iterations):
        size_len = len(A)
        C = [[0] * size_len for _ in range(size_len)]
        for i in range(size_len):
            for j in range(size_len):
                for k in range(size_len):
                    C[i][j] += A[i][k] * B[k][j]
        A = C # Reuse

    end_time = time.time()
    return end_time - start_time

def gpu_benchmark():
    """
    Performs ACTUAL GPU matrix multiplication using PyTorch.
    """
    if not TORCH_AVAILABLE:
        return 1.0, "Placeholder (No Torch)"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dev_name = "Apple Silicon (MPS)"
    else:
        device = torch.device("cpu")
        dev_name = "CPU (Torch Fallback)"

    print(f"[*] Initializing VRAM for {dev_name} with {GPU_MATRIX_SIZE}x{GPU_MATRIX_SIZE} Float32 matrix...")
    
    try:
        t1 = torch.rand(GPU_MATRIX_SIZE, GPU_MATRIX_SIZE, device=device)
        t2 = torch.rand(GPU_MATRIX_SIZE, GPU_MATRIX_SIZE, device=device)
        
        # Warmup
        _ = torch.matmul(t1, t2)
        
        start = time.time()
        for _ in range(GPU_ITERATIONS):
            res = torch.matmul(t1, t2)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
                
        end = time.time()
        duration = end - start
        return duration, dev_name

    except Exception as e:
        print(f"[!] GPU Test Failed: {e}")
        return 1.0, "Failed"

def threaded_prime_stress(n, num_threads):
    """Benchmarks multithread performance."""
    results = []
    def thread_target():
        duration = calculate_primes(n) 
        results.append(duration)

    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=thread_target)
        threads.append(thread)

    start_time = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    end_time = time.time()
    
    return end_time - start_time

def export_to_database(data):
    """
    Exports the benchmark results to a PostgreSQL database.
    """
    if not DB_AVAILABLE:
        print("[!] Psycopg2 not loaded. Skipping DB export.")
        return

    # Retrieve Environment Variables
    host = os.environ.get('DB_HOST')
    user = os.environ.get('POSTGRES_USER')
    password = os.environ.get('POSTGRES_PASSWORD')
    dbname = os.environ.get('POSTGRES_DB')

    if not all([host, user, password, dbname]):
        print("-" * 40)
        print("[!] Missing DB environment variables. Skipping export.")
        print("[!] Ensure DB_HOST, POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB are set.")
        print("-" * 40)
        return

    print(f"[*] Connecting to database {dbname} at {host}...")

    try:
        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            dbname=dbname
        )
        cur = conn.cursor()

        # Create Table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            version VARCHAR(50),
            os_system VARCHAR(255),
            processor VARCHAR(255),
            architecture VARCHAR(50),
            logical_cores INT,
            gpu_device VARCHAR(255),
            score_single_thread FLOAT,
            score_matrix FLOAT,
            score_multithread FLOAT,
            score_gpu FLOAT,
            score_total FLOAT
        );
        """
        cur.execute(create_table_query)

        # Insert Data
        insert_query = """
        INSERT INTO benchmark_results 
        (version, os_system, processor, architecture, logical_cores, gpu_device, 
         score_single_thread, score_matrix, score_multithread, score_gpu, score_total)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        cur.execute(insert_query, (
            data['version'],
            data['os'],
            data['processor'],
            data['arch'],
            data['cores'],
            data['gpu_dev'],
            data['score_single'],
            data['score_matrix'],
            data['score_multi'],
            data['score_gpu'],
            data['score_total']
        ))

        conn.commit()
        cur.close()
        conn.close()
        print("[+] Results successfully exported to database table 'benchmark_results'.")
        print("-" * 40)

    except Exception as e:
        print(f"[!] Database error: {e}")
        print("-" * 40)


def run_benchmark():
    """Runs all benchmarks, scores them, and exports to DB."""
    processor_name, architecture = get_cpu_info()
    gpu_name = get_gpu_info()
    num_cores = os.cpu_count() or 1
    
    sys_os_str = f"{platform.system()} {platform.release()}"

    print("--- Performance Benchmark (Python + Torch + Postgres) ---")
    print(f"System OS: {sys_os_str}")
    print("-" * 40)
    print(f"Processor Type: {processor_name}")
    print(f"Architecture: {architecture}")
    print(f"Logical Cores: {num_cores}")
    print(f"GPU Info: {gpu_name}")
    print("-" * 40)
    
    # --- Benchmarks ---
    print(f"[*] Single-Thread Test...")
    prime_time = calculate_primes(PRIME_TARGET_NUMBER)
    
    print(f"[*] Matrix Float Test...")
    matrix_time = matrix_benchmark(MATRIX_SIZE, MATRIX_ITERATIONS)
    
    print(f"[*] Multithread Test ({num_cores} cores)...")
    multi_thread_time = threaded_prime_stress(PRIME_TARGET_NUMBER, num_cores)
    
    print(f"[*] GPU Compute Test...")
    gpu_time, actual_gpu_device = gpu_benchmark()
    
    # --- Scoring ---
    BASE_PRIME_TIME = 0.5   
    BASE_MATRIX_TIME = 15.0 
    BASE_MULTI_TIME = 0.6   
    BASE_GPU_TIME = 2.0     

    prime_score = (BASE_PRIME_TIME / prime_time) * 1000
    matrix_score = (BASE_MATRIX_TIME / matrix_time) * 1000
    multithread_score = (BASE_MULTI_TIME / multi_thread_time) * 1000
    
    if not TORCH_AVAILABLE or "Placeholder" in actual_gpu_device:
        gpu_score = 100 
    else:
        gpu_score = (BASE_GPU_TIME / gpu_time) * 1000
    
    final_cpu_score = (prime_score * 0.25) + (matrix_score * 0.25) + \
                      (multithread_score * 0.25) + (gpu_score * 0.25)

    print("\n--- Results ---")
    print(f"Single-Thread Score: {prime_score:.0f}")
    print(f"Matrix Float Score:  {matrix_score:.0f}")
    print(f"Multithread Score:   {multithread_score:.0f}")
    print(f"GPU Compute Score:   {gpu_score:.0f}")
    print("---------------------------------")
    print(f"OVERALL SYSTEM SCORE: {final_cpu_score:.0f}")
    print("---------------------------------")
    
    # --- Data Collection ---
    benchmark_data = {
        'version': BENCHMARK_VERSION,
        'os': sys_os_str,
        'processor': processor_name,
        'arch': architecture,
        'cores': num_cores,
        'gpu_dev': actual_gpu_device,
        'score_single': prime_score,
        'score_matrix': matrix_score,
        'score_multi': multithread_score,
        'score_gpu': gpu_score,
        'score_total': final_cpu_score
    }

    # --- Export ---
    export_to_database(benchmark_data)
    
    print(f"\nBenchmark Version: {BENCHMARK_VERSION}")

if __name__ == "__main__":
    run_benchmark()