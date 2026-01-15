import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class ConstrainedAttentionModel(nn.Module):
    def __init__(self, vocab_size, init_scale: float = 0.0, order_k=2, mask_first: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.k = order_k
        self.mask_first = mask_first
        
        # Interaction Matrix C of shape (k, k)
        if init_scale > 0.0:
            # Initialize with small Gaussian noise scaled by init_scale
            self.params = nn.Parameter(torch.randn(order_k, order_k) * init_scale)
        else:
            # Initialize to exact zeros (Uniform attention behavior at start)
            self.params = nn.Parameter(torch.zeros(order_k, order_k))

    def get_windows(self, x):
        """
        Creates a sliding window representation of the sequence.
        Output: (Batch, T, k, V)
            [:, t, 0, :] = x_t
            [:, t, 1, :] = x_{t-1}
            ...
        """
        B, T = x.shape
        emb = F.one_hot(x, num_classes=self.vocab_size).float() # (B, T, V)
        
        # Pad the sequence with k-1 zeros at the beginning to handle history for early tokens
        # Padding shape: (B, k-1, V)
        padding = torch.zeros(B, self.k - 1, self.vocab_size, device=x.device)
        padded_emb = torch.cat([padding, emb], dim=1) # (B, T + k - 1, V)
        
        # Unfold to create windows
        # Dimension 1 is time. Size is k. Step is 1.
        # This creates (B, T, V, k) -> we permute to (B, T, k, V)
        # Note: Unfold behavior: (B, T_out, V, k)
        windows = padded_emb.unfold(dimension=1, size=self.k, step=1)
        windows = windows.permute(0, 1, 3, 2)
        
        # Reverse the k dimension so index 0 is Current (x_t) and index k-1 is Oldest
        # Unfold naturally gives [x_{t-k+1}, ..., x_t]. We want [x_t, ..., x_{t-k+1}]
        windows = windows.flip(dims=[2])
        
        return windows

    def forward(self, x):
        B, T = x.shape
        
        # 1. Create Windows: (B, T, k, V)
        windows = self.get_windows(x)
        
        # 2. Define Query and Key
        # Query: The window at the very last position T. Shape (B, k, V)
        q_window = windows[:, -1, :, :] 
        
        # Key: The windows at all positions. Shape (B, T, k, V)
        k_windows = windows
        
        if self.mask_first:
            k_windows = k_windows[:, self.k-1:, :, :]

        # 3. Calculate Scores using Einsum
        # Equation: Score[b, t] = Sum_{i,j} ( Q[b, i, v] * K[b, t, j, v] * Params[i, j] )
        # b: batch
        # t: time (keys)
        # i: lag index for Query
        # j: lag index for Key
        # v: vocab dimension
        
        # First, calculate dot products between all lag pairs: (B, T, k_q, k_k)
        # Contract over 'v'
        match_matrix = torch.einsum('b i v, b t j v -> b t i j', q_window, k_windows)
        
        # Second, weight these matches by the parameter matrix C: (k, k)
        # Contract over 'i' and 'j'
        scores = torch.einsum('b t i j, i j -> b t', match_matrix, self.params)
        
        # 4. Causal Masking (Mask out the query position itself)
        mask = torch.ones_like(scores)
        mask[:, -1] = 0 
        scores = scores.masked_fill(mask == 0, -1e9)

        # 5. Attention Probabilities
        attn_weights = F.softmax(scores, dim=-1) # (B, T)
        
        # 6. Value Aggregation
        # Value is usually just the token at x_t (lag 0 of the key window)
        values = k_windows[:, :, 0, :] # (B, T, V)
        
        # (B, T) x (B, T, V) -> (B, V)
        output = torch.matmul(attn_weights.unsqueeze(1), values).squeeze(1)
        
        return output

def generate_dirichlet_markov_data(batch_size, seq_len, vocab_size, order, alpha):
    """
    Generates sequences from an Order-K Markov Chain.
    P(x_t | x_{t-1}, ..., x_{t-k})
    """
    # 1. Create Transition Tensor
    # Shape: (V, V, ..., V) with order+1 dimensions.
    # The indices [x_{t-k}, ..., x_{t-1}] point to the prob dist for x_t.
    shape = [vocab_size] * (order + 1)
    num_states = vocab_size ** order
    
    # Flatten to sample Dirichlet, then reshape
    flat_transitions = torch.distributions.Dirichlet(torch.full((vocab_size,), alpha)).sample((num_states,))
    transitions = flat_transitions.view(*shape)
    
    data = []
    targets = []
    
    for _ in range(batch_size):
        # Initialize with 'order' random tokens
        seq = [torch.randint(0, vocab_size, (1,)).item() for _ in range(order)]
        
        # Generate sequence
        # We generate enough tokens to fill seq_len + 1 target
        gen_len = seq_len if seq_len > order else order + 1
        
        for _ in range(gen_len):
            # Get history tuple: last 'order' tokens
            # transitions[t1, t2, ...] returns probs
            history = tuple(seq[-order:]) if order > 0 else ()
            probs = transitions[history]
            
            next_token = torch.multinomial(probs, 1).item()
            seq.append(next_token)
            
        # Slice to return fixed length inputs
        # We take the last seq_len tokens as input x
        full_seq_tensor = torch.tensor(seq)
        
        x = full_seq_tensor[-(seq_len+1):-1]
        y = full_seq_tensor[-1]
        
        data.append(x)
        targets.append(y)
        
    return torch.stack(data), torch.stack(targets)

def evaluate_model(model, x_val, y_val):
    """
    Evaluates the model on a fixed validation set.
    """
    model.eval() # Switch to eval mode
    with torch.no_grad():
        probs = model(x_val)
        # Numerical stability handling
        loss = F.nll_loss(
            torch.log(probs.clamp(min=1e-9)), 
            y_val
        )
    model.train() # Switch back to train mode
    return loss.item()

def main():
    parser = argparse.ArgumentParser(description="Train Constrained Attention on Markov Chains")
    
    # Model & Data Params
    parser.add_argument("--vocab_size", type=int, default=5, help="Size of vocabulary")
    parser.add_argument("--seq_len", type=int, default=50, help="Length of context sequence")
    parser.add_argument("--order_k", type=int, default=1, help="Order of the history window (k)")
    parser.add_argument("--mask_first", action="store_true", help="Whether to mask the first token in attention")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration parameter")
    parser.add_argument("--init_scale", type=float, default=0.0, help="Initialization scale")

    # Training Params
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_sub_pattern", action="store_true", help="Only train sub pattern, keep others at 0")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between printing training loss")
    parser.add_argument("--eval_interval", type=int, default=500, help="Steps between running validation")
    parser.add_argument("--val_size", type=int, default=5000, help="Fixed validation set size")

    # WandB Params
    parser.add_argument("--wandb_project", type=str, default="induction-head-markov", help="WandB Project Name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB Entity (User/Team) Name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )

    # Setup
    model = ConstrainedAttentionModel(args.vocab_size, init_scale=args.init_scale, order_k=args.order_k + 1, mask_first=args.mask_first)
    if args.train_sub_pattern:
            print(f"Configuration: Training ONLY induction diagonals. All else 0.")
            
            # Create mask of shape (model_k, model_k)
            mask = torch.zeros(args.order_k + 1, args.order_k + 1)
            
            # Activate the induction diagonal: Query Lag i matches Key Lag i+1
            # Example: Q0 matches K1 (Bigram), Q1 matches K2 (Trigram), etc.
            for i in range(args.order_k):
                mask[i, i+1] = 1.0
                
            print(f"Gradient Mask:\n{mask}")
            model.params.register_hook(lambda grad: grad * mask.to(grad.device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Initial Params: {model.params.data}")

    # --- Generate Fixed Validation Set ---
    print(f"Generating fixed validation set (size {args.val_size})...")
    x_val, y_val = generate_dirichlet_markov_data(
        args.val_size, args.seq_len, args.vocab_size, args.order_k, args.alpha
    )

    # Training Loop
    for step in range(args.steps):
        # Generate fresh batch
        x_batch, y_batch = generate_dirichlet_markov_data(
            args.batch_size, args.seq_len, args.vocab_size, args.order_k, args.alpha
        )
        
        optimizer.zero_grad()
        probs = model(x_batch)
        loss = F.nll_loss(
            torch.log(probs.clamp(min=1e-9)), 
            y_batch
        )
        loss.backward()
        optimizer.step()
        
        # Logging
        if step % args.log_interval == 0 or step % args.eval_interval == 0:
            # --- 1. Prepare Data ---
            matrix_data = model.params.detach().cpu().numpy()
            matrix_rounded = np.around(matrix_data, 3)
            
            # --- 2. Create Plot using Matplotlib ---
            # This is robust and works on all WandB versions
            fig, ax = plt.subplots(figsize=(5, 5))
            cax = ax.matshow(matrix_rounded, cmap='viridis')
            
            # Add text (values) inside the squares
            for i in range(args.order_k + 1):
                for j in range(args.order_k + 1):
                    ax.text(j, i, str(matrix_rounded[i, j]), 
                            ha="center", va="center", color="white", fontsize=8)
            
            # Labels
            labels = [f"L{i}" for i in range(args.order_k + 1)]
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            
            # Label Axes (Recall: Rows=Query, Cols=Key)
            plt.xlabel("Key Lag")
            plt.ylabel("Query Lag")
            plt.title(f"Parameter Matrix (Step {step})")
            
            # --- 3. Log to WandB ---
            log_dict = {
                "train_loss": loss.item(), 
                "step": step,
                "parameter_matrix": wandb.Image(fig)
            }

            # Close plot to prevent memory leaks
            plt.close(fig)

            # --- LOG FULL MATRIX ---
            # Iterates through every cell in the k x k matrix
            # C[i, j] = Weight for Query Lag i interacting with Key Lag j
            for i in range(args.order_k + 1):
                for j in range(args.order_k + 1):
                    val = model.params[i, j].item()
                    # Key format: "C_q{row}_k{col}"
                    log_dict[f"param/C_q{i}_k{j}"] = val

            # Print a quick summary to the console (just the induction diagonal)
            # We assume the "Induction Head" logic lies on the +1 diagonal (j = i + 1)
            induction_vals = []
            for i in range(args.order_k):
                induction_vals.append(f"{model.params[i, i+1].item():.2f}")
                
            print(f"Step {step}: Loss = {loss.item():.4f} | Induction Diags: {induction_vals}")

            if step % args.eval_interval == 0:
                val_loss = evaluate_model(model, x_val, y_val)
                print(f"    >>> EVAL: Val Loss = {val_loss:.4f}")
                print(f"    Param Matrix:\n{model.params.data}")
                log_dict.update({"val_loss": val_loss})

            if not args.no_wandb:
                wandb.log(log_dict)

    # Final Result
    print("-" * 30)
    print("Optimization Complete.")
    print(f"Final Parameters: {model.params.data}")
    
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()