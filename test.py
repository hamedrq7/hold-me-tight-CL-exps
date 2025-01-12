import numpy as np 
import matplotlib.pyplot as plt 

# EPOCHS = 30
# MAX_LR = 0.21
# lr_schedule = lambda t: np.interp([t], [0, EPOCHS * 2 // 5, EPOCHS], [0, MAX_LR, 0])[0]  # Triangular (cyclic) learning rate schedule

# Simulated batches in an epoch
trainloader = range(60000//128)  # 100 batches per epoch

# Lambda function
EPOCHS = 100
MAX_LR = 0.01
gamma = 0.1
milestones = [55, 75, 90]
lr_schedule = lambda t: MAX_LR * (gamma ** sum([int(t) >= milestone for milestone in milestones]))

learning_rates = []
# Simulate training loop
for epoch in range(EPOCHS):
    for batch_idx in range(len(trainloader)):
        t = epoch + (batch_idx + 1) / len(trainloader)  # Fractional epoch
        lr = lr_schedule(t)
        print(f"Epoch {t:.2f}: Learning Rate = {lr}")

        learning_rates.append((t, lr))  # Store epoch and learning rate

# Extract time (epochs) and learning rates
epochs, lrs = zip(*learning_rates)

# Plot learning rates
plt.figure(figsize=(10, 6))
plt.plot(epochs, lrs, label="Learning Rate")
plt.xlabel("Epoch (including fractional values)")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.legend()
plt.show()