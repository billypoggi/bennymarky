BennyMarky 📊

BennyMarky is a containerized Python application designed to perform high-load benchmarking on CPU and GPU resources. It executes computational stress tests (matrix multiplications, floating-point operations) and persists the performance metrics into a PostgreSQL database.

🚀 Features

CPU Stress Test: Multi-core floating-point arithmetic and prime number generation.

GPU Stress Test: CUDA-accelerated matrix multiplications (requires NVIDIA GPU).

Persistence: Automatically saves benchmark results (Timestamp, TFLOPS, Memory Usage, Duration) to a dedicated database.

Dockerized: runs in isolated containers for consistent environment reproduction.

🛠 Prerequisites

Before running this project, ensure you have the following installed:

Docker Desktop or Docker Engine

NVIDIA Drivers (if running GPU tests)

NVIDIA Container Toolkit: Required for Docker containers to access your host GPU.

Linux: sudo apt-get install -y nvidia-container-toolkit

Windows: Included in Docker Desktop (ensure WSL2 backend is selected).

📂 Project Structure

bennymarky/
├── Dockerfile              # Blueprint for the Python App
├── docker-compose.yml      # Orchestration for App + DB
├── requirements.txt        # Python dependencies
├── src/
│   └── main.py             # Entry point (your script)
└── README.md


⚙️ Configuration

Create a .env file in the root directory (optional, defaults are set in docker-compose):

DB_NAME=benchmark_db
DB_USER=default_admin
DB_PASSWORD=default_password
DB_HOST=db


🏗️ Installation & Usage

1. Build and Run the Stack

This command pulls the Database image and builds your Python application image.

docker-compose up --build -d


2. Check Container Status

Ensure both the database and the app are running:

docker ps


3. Run a Benchmark

Since the app container is running in the background, you can execute the benchmark script inside it:

# Run the main benchmarking script
docker-compose exec benchmark-app python src/main.py


4. View Logs

To see the output of the application or database:

docker-compose logs -f benchmark-app


5. Shutdown

To stop containers and remove the network (data persists in the docker volume):

docker-compose down


🖥 GPU Passthrough Note

The docker-compose.yml is configured to request GPU resources. If you do not have a GPU or the NVIDIA Toolkit installed, you may need to comment out the deploy section in docker-compose.yml to run CPU-only tests.
