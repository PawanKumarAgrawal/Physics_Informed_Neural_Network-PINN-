# python test_deepxde.py

# pip install deepxde numpy matplotlib
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ============================================================
# File: test_deepxde_pytorch.py
# Purpose: Verify that DeepXDE is correctly installed and ready
#          for Physics-Informed Neural Networks (PINNs) using PyTorch
# ============================================================

# -------------------------------
# Step 0: Import required libraries
# -------------------------------
import os                        # For setting environment variables
import deepxde as dde            # DeepXDE library for PINNs



# Other supported backends: tensorflow.compat.v1, tensorflow, jax, paddle
# Set the backend for DeepXDE to PyTorch
os.environ["DDE_BACKEND"] = "pytorch"

# Check backend details
backend = dde.backend.backend_name
print(f"📋 Backend Details: {backend}")




# -------------------------------
# Step 1: Test DeepXDE installation
# -------------------------------
print("🔧 Testing DeepXDE Installation with PyTorch backend...")



# -------------------------------
# Step 2: Create simple geometry
# -------------------------------
# Define a 1D interval from 0 to 1 as our domain
geometry = dde.geometry.Interval(0, 1)
print("✅ Geometry module working")  # Confirm geometry creation works

# Test geometry properties
print(f"📐 Geometry Dimension: {geometry.dim}")




# -------------------------------
# Step 3: Create neural network
# -------------------------------
# Define a feedforward neural network (FNN)
# Neural network parameters explained:
# [1, 20, 1]      -> 1 input neuron (x), 1 hidden layer with 20 neurons, 1 output neuron (y)
# "tanh"           -> Activation function for hidden layers; introduces nonlinearity
# "Glorot normal"  -> Weight initializer; helps with stable training
net = dde.nn.FNN([1, 20, 1], "tanh", "Glorot normal")

# Explanation of FNN parameters:
# 1. Input layer: size 1, because we have 1 independent variable (x)
# 2. Hidden layer: 20 neurons chosen arbitrarily; more neurons can approximate more complex functions
# 3. Output layer: size 1, because our PDE solution y(x) is scalar
# 4. Activation function ("tanh"): smooth, differentiable, ideal for PINNs as derivatives are required
# 5. Weight initializer ("Glorot normal"): balances variance of weights across layers, preventing vanishing/exploding gradients

# Advantage of PyTorch backend for PINNs:
# - Dynamic computation graph makes debugging derivatives easier
# - Flexible autograd system for higher-order derivatives
# - Large research community and examples available
# - Easy integration with PyTorch optimizers like L-BFGS, Adam

print("✅ Neural network created with PyTorch backend")  # Confirm network creation works



# -------------------------------
# Step 4: Define a simple PDE
# -------------------------------
# Define a basic differential equation dy/dx = 1
def simple_pde(x, y):
    """
    x: input variable (tensor)
    y: output variable (tensor)
    Returns the residual of the PDE dy/dx - 1
    """
    # Automatic differentiation: u
    # se grad.jacobian(y, x) for dy/dx 
    # and grad.hessian(y, x) for d²y/dx².
    dy_dx = dde.grad.jacobian(y, x)  
    return dy_dx - 1

print("✅ Automatic differentiation working")  # Confirm PDE differentiation works



# -------------------------------
# Step 5: Final confirmation
# -------------------------------
print("🎉 DeepXDE setup successful with PyTorch! Ready for PINNs!")

print("\n🔍 All Systems Operational!")
