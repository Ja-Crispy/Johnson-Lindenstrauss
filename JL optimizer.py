import torch
import logging
import sys

# Configure logging to output to both terminal and file
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("output.log"),  # Log to file
        logging.StreamHandler(sys.stdout)  # Log to terminal explicitly
    ]
)

# Set computation device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_steps = 50000

for vector_len in [2, 3, 4, 7, 10, 30, 100, 300, 1000, 3000, 10000]:
    for num_vectors in [10, 30, 100, 300, 1000, 3000, 10000, 20000, 30000]:

        loss_exp = min(int(60 / torch.log(torch.tensor(vector_len, dtype=torch.float32))), 20)
        step_now = 0

        # Create and normalize big_matrix on GPU
        big_matrix = torch.randn(num_vectors, vector_len, device=device, dtype=torch.float32)
        big_matrix = torch.nn.functional.normalize(big_matrix, p=2, dim=1)  # More stable normalization
        big_matrix.requires_grad_(True)

        # Set up an Optimization loop to create nearly-perpendicular vectors
        optimizer = torch.optim.Adam([big_matrix], lr=0.01)
        big_id = torch.eye(num_vectors, device=device, dtype=torch.float32)  # Identity matrix on GPU
        c = 10  # Initial value for c


        while vector_len < num_vectors:
            optimizer.zero_grad()

            # Normalize big_matrix rows
            big_matrix_norm = torch.nn.functional.normalize(big_matrix, p=2, dim=1)
            dot_products = torch.matmul(big_matrix_norm, big_matrix_norm.T)

            # Punish deviation from orthogonality
            diff = dot_products - big_id
            loss = (diff.abs()**loss_exp).sum()

            # Compute c using PyTorch instead of NumPy
            epsilon = diff.max().item()
            c = min(vector_len / torch.log(torch.tensor(num_vectors, device=device, dtype=torch.float32)) * epsilon**2, c)
            step_now += 1

            if step_now == num_steps:
                log_msg = f"Dimensions: {vector_len}   Vectors: {num_vectors}   Lowest C: {c}  Steps: {step_now}"
                print(log_msg, flush=True)  # Force terminal output
                logging.info(log_msg)  # Logs to both file and terminal
                sys.stdout.flush()  # Ensure output is printed immediately
                break
            if step_now % 100 == 0: # Print updates at regular intervals
                log_msg = f"Dimensions: {vector_len}   Vectors: {num_vectors}   Lowest C: {c}  Steps: {step_now}"
                print(log_msg, flush=True)  # Force terminal output
                logging.info(log_msg)  # Logs to both file and terminal
                sys.stdout.flush()  # Ensure output is printed immediately

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
