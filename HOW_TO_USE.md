
# How to Use the Optimiser API

This guide provides a complete, runnable example to get you started with the optimiser API.

---

## 1. Project Structure

First, ensure your project is structured correctly.  
The example script below (`your_training_script.py`) should be in your project's root directory.

```

your_project/
├── tools/
│   ├── opt_parent.py
│   ├── Utilities.py
│   └── ...
├── Config/
│   └── RegisterDecorator.py
├── Optimisers/
│   ├── **init**.py
│   ├── SGD.py
│   ├── NAG.py
│   └── Adam.py
└── your_training_script.py

````

---

## 2. Complete Example

The following is a self-contained script that you can copy and paste into `your_training_script.py`.  
It demonstrates how to define parameters, build an optimiser, and run a basic training loop.

```python
import numpy as np
# Make sure to adjust the import path based on your project structure
from tools import build


# --- 1. Defining Parameters ---
# Your model's parameters need a `.data` attribute (its value) and
# a `.grad` attribute (for storing gradients).
# If you are using PyTorch, your model's parameters (`model.parameters()`)
# will already work. For NumPy, you can use this simple class.

class Parameter:
    """A simple class to hold parameter data and its gradient."""

    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        self.data = data
        self.grad = np.zeros_like(self.data)


# Example: Create some parameters for a hypothetical model
param1 = Parameter([5.0, -2.0])
param2 = Parameter([1.0])
model_params = [param1, param2]

# --- 2. Building the Optimiser ---
# Use the `build` function to create an optimiser instance.
# Pass the name of the optimiser, a list of your model's parameters,
# and any required hyperparameters.

optimizer = build(
    "SGD",
    model_params,
    lr=0.01,
    weight_decay=0.001
)

# You could also build a different optimiser, like Adam:
# optimizer = build(
#     "Adam",
#     model_params,
#     lr=0.001,
#     weight_decay=0.0,
#     beta1=0.9,
#     beta2=0.999
# )

print("--- Initial Parameters ---")
print(f"Param 1: {model_params[0].data}")
print(f"Param 2: {model_params[1].data}")
print("-" * 26)

# --- 3. The Training Loop ---
# In a real training loop, you would calculate loss and gradients
# using your model and data. Here, we simulate this process.

print("\n--- Starting Training Loop ---\n")
for epoch in range(5):
    # a. Clear old gradients from the previous step
    optimizer.zero_grad()

    # b. Simulate computing gradients (e.g., from a loss function)
    # In a real scenario, this would be your backpropagation step.
    model_params[0].grad = np.array([0.5, -0.1]) * (epoch + 1)
    model_params[1].grad = np.array([-0.02]) * (epoch + 1)

    # c. Update the parameters using the optimiser
    optimizer.step()

    # d. Print updated parameters to see the change
    print(f"--- Epoch {epoch + 1} ---")
    print(f"  Param 1 Grad: {model_params[0].grad}")
    print(f"  Updated Param 1: {model_params[0].data}")
    print(f"  Param 2 Grad: {model_params[1].grad}")
    print(f"  Updated Param 2: {model_params[1].data}")
    print("-" * 20)
````
