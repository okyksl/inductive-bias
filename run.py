import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.special

class ConstrainedAttentionModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        order_k=2,
        mask_first: bool = False,
        init_scale: float = 0.0,
        couple_diagonals: bool = False,  # Changed argument
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.k = order_k
        self.mask_first = mask_first
        self.couple_diagonals = couple_diagonals

        # Precompute indices for diagonal coupling (Toeplitz structure)
        # We need this to map the (k, k) matrix to the diagonal parameters
        if self.couple_diagonals:
            # Range of offsets: -(k-1) to +(k-1). Total 2k-1 parameters.
            self.num_diags = 2 * order_k - 1

            # Create indices matrix: indices[i, j] = j - i + (k - 1)
            # This maps matrix coordinates to the flat parameter vector index
            rows = torch.arange(order_k).unsqueeze(1)
            cols = torch.arange(order_k).unsqueeze(0)
            self.register_buffer("diag_indices", cols - rows + (order_k - 1))

            # Initialize parameters for diagonals
            if init_scale > 0.0:
                self.diag_params = nn.Parameter(
                    torch.randn(self.num_diags) * init_scale
                )
            else:
                self.diag_params = nn.Parameter(torch.zeros(self.num_diags))
        else:
            # === FULL MATRIX MODE ===
            if init_scale > 0.0:
                self.params = nn.Parameter(torch.randn(order_k, order_k) * init_scale)
            else:
                self.params = nn.Parameter(torch.zeros(order_k, order_k))

    def get_windows(self, x):
        B, T = x.shape
        emb = F.one_hot(x, num_classes=self.vocab_size).float()  # (B, T, V)

        # Pad the sequence with k-1 zeros at the beginning to handle history for early tokens
        # Padding shape: (B, k-1, V)
        padding = torch.zeros(B, self.k - 1, self.vocab_size, device=x.device)
        padded_emb = torch.cat([padding, emb], dim=1)  # (B, T + k - 1, V)

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

    def get_effective_params(self):
        """Returns the (k, k) matrix C used for attention scores."""
        if self.couple_diagonals:
            # Construct the matrix from the diagonal parameters
            # self.diag_params is (2k-1,)
            # self.diag_indices is (k, k)
            return self.diag_params[self.diag_indices]
        else:
            return self.params

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
            k_windows = k_windows[:, self.k - 1 :, :, :]

        # 3. Calculate Scores using Einsum
        # Equation: Score[b, t] = Sum_{i,j} ( Q[b, i, v] * K[b, t, j, v] * Params[i, j] )
        # b: batch
        # t: time (keys)
        # i: lag index for Query
        # j: lag index for Key
        # v: vocab dimension

        # First, calculate dot products between all lag pairs: (B, T, k_q, k_k)
        # Contract over 'v'
        match_matrix = torch.einsum("b i v, b t j v -> b t i j", q_window, k_windows)

        # Retrieve effective C matrix (either full or Toeplitz)
        C = self.get_effective_params()

        # Contract over 'i' and 'j'
        scores = torch.einsum("b t i j, i j -> b t", match_matrix, C)

        # 4. Causal Masking (Mask out the query position itself)
        mask = torch.ones_like(scores)
        mask[:, -1] = 0
        scores = scores.masked_fill(mask == 0, -1e9)

        # 5. Attention Probabilities
        attn_weights = F.softmax(scores, dim=-1)  # (B, T)

        # 6. Value Aggregation
        # Value is just the token at x_t (lag 0 of the key window)
        values = k_windows[:, :, 0, :]  # (B, T, V)

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
    num_states = vocab_size**order

    # Flatten to sample Dirichlet, then reshape
    flat_transitions = torch.distributions.Dirichlet(
        torch.full((vocab_size,), alpha)
    ).sample((num_states,))
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

        x = full_seq_tensor[-(seq_len + 1) : -1]
        y = full_seq_tensor[-1]

        data.append(x)
        targets.append(y)

    return torch.stack(data), torch.stack(targets)


def get_counts_and_probs(seq, target, order, vocab_size, alpha):
    """
    Computes the Bayes optimal probability for the specific target token
    given the sequence history and Dirichlet prior.
    """
    # 1. Identify the Context (Query)
    # The context is the last 'order' tokens of the sequence
    if order > 0:
        context = seq[-order:]
    else:
        context = torch.tensor([], device=seq.device)

    # 2. Scan History for Matches
    # We look at the sequence up to T-1 to see what followed this context previously
    # Unfold sequence into windows of size (order + 1)
    # e.g. if seq=[A, B, C, A, B], order=1. Context=B.
    # Windows: [A, B], [B, C], [C, A], [A, B]
    # Matches: [A, B] (prev), [A, B] (current).
    # We only care about PAST occurrences, so we look at seq[:-1]

    # We need to count occurrences of (Context -> v) in x_1...x_T
    # Construct all windows of size order+1 in the sequence x
    # seq shape: (T,)

    # If sequence is shorter than order+1, we rely purely on prior
    if seq.shape[0] < order + 1:
        # Prior probability
        return (alpha) / (vocab_size * alpha)

    # Unfold: (num_windows, order+1)
    windows = seq.unfold(0, order + 1, 1)

    # The last window in 'windows' is the one ending at T (the query itself).
    # We want to count occurrences in the history (indices 0 to T-2)
    # Actually, we are predicting x_{T+1}. The input x includes x_1...x_T.
    # We want to find times where context x_{T-order+1}...x_T appeared previously.
    # So we look at windows in x[:-1] that match x[-order:]

    # Let's simplify:
    # Query Context: seq[-order:]
    # Search Space: seq (excluding the implied future)

    # We want to count N(context -> v) within the observed sequence `seq`
    # Note: The standard "Induction Head" task is usually:
    # "Predict next token based on previous occurrence in THIS context."

    # Scan windows
    # windows[:, :-1] are the contexts
    # windows[:, -1] are the next tokens

    # We exclude the very last position because that is "now" (we don't know next token yet)
    # Actually, the 'seq' passed here is x_batch (inputs).
    # The 'target' is y_batch (the true next token).
    # So we scan `seq` entirely.

    current_context = seq[-order:] if order > 0 else torch.empty(0, device=seq.device)

    # Find matches
    if order > 0:
        # Compare all contexts in the sequence with current_context
        # windows shape: (num_windows, order+1)
        # We compare windows[:, :-1] with current_context
        history_windows = windows  # All transitions observed so far

        # Check equality
        # (num_windows, order) == (order) -> (num_windows, order)
        # All must match along dim 1
        matches = (history_windows[:, :-1] == current_context).all(dim=1)

        # Get the next tokens for the matches
        next_tokens = history_windows[matches, -1]
    else:
        # Zero order: just count all tokens
        next_tokens = seq

    # 3. Compute Dirichlet Posterior Probability
    # Count occurrences of the specific target
    count_target = (next_tokens == target).sum().item()
    total_counts = next_tokens.shape[0]

    return (count_target + alpha) / (total_counts + vocab_size * alpha)


def compute_optimal_loss(x_batch, y_batch, vocab_size, order, alpha):
    """
    Computes the theoretical lower bound loss (Bayes Optimal) for a batch.
    """
    nll_sum = 0.0

    # We iterate because vectorizing variable-length matching is tricky
    # and this is only for validation/logging (speed is less critical)
    for i in range(x_batch.shape[0]):
        seq = x_batch[i]
        target = y_batch[i]

        prob = get_counts_and_probs(seq, target, order, vocab_size, alpha)

        # Clamp for numerical stability
        prob = max(prob, 1e-9)
        nll_sum += -np.log(prob)

    return nll_sum / x_batch.shape[0]

def estimate_overlap_distribution(
    data_gen_fn, 
    batch_size, 
    seq_len, 
    vocab_size, 
    order, 
    alpha, 
    max_k, 
    mask_first=False, 
    variance_threshold=0.01, 
    min_samples=100, 
    max_samples=10000
):
    # Effective context length (excluding x_t)
    effective_k = max_k - 1
    if effective_k < 1:
        return {}

    # Precompute Binomial Coefficients C[j, i] = (j choose i)
    # Shape: (effective_k+1, effective_k+1)
    binom_coeffs = torch.zeros((effective_k + 1, effective_k + 1), dtype=torch.float32)
    for j in range(effective_k + 1):
        for i in range(j + 1):
            binom_coeffs[j, i] = scipy.special.comb(j, i)

    # Storage for all raw samples (safest for variance calc)
    # N_stats[j] will store list of counts for exactly j matches
    # K_stats[i] will store list of counts for derived variable K_i
    N_samples = {j: [] for j in range(effective_k + 1)}
    K_samples = {i: [] for i in range(effective_k + 1)}
    
    total_samples = 0
    
    while total_samples < max_samples:
        x_batch, _ = data_gen_fn(batch_size, seq_len, vocab_size, order, alpha)
        B, T = x_batch.shape
        
        # --- 1. Get Windows ---
        if mask_first:
            if T < max_k: continue 
            windows = x_batch.unfold(dimension=1, size=max_k, step=1)
        else:
            padding = torch.zeros(B, max_k - 1, dtype=x_batch.dtype, device=x_batch.device)
            x_padded = torch.cat([padding, x_batch], dim=1) 
            windows = x_padded.unfold(dimension=1, size=max_k, step=1)
            
        context_windows = windows[:, :, :-1] # Slice off current token
        query_ctx = context_windows[:, -1, :] 
        history_ctx = context_windows[:, :-1, :] 
        
        if history_ctx.shape[1] == 0: continue

        # --- 2. Compute N_j (Overlap Counts) ---
        # Matrix of matches: (B, History)
        # Sum over token dim to get overlap count for each history window
        overlaps = (history_ctx == query_ctx.unsqueeze(1)).sum(dim=2) # (B, H)
        
        # We need the vector [N_0, N_1, ..., N_k] for EACH sequence in batch
        # shape: (B, effective_k + 1)
        # We can use scatter_add or a simple loop since effective_k is small
        N_counts_batch = torch.zeros((B, effective_k + 1), device=x_batch.device)
        
        for j in range(effective_k + 1):
            # Count how many history windows have overlap == j
            count_j = (overlaps == j).float().sum(dim=1) # (B,)
            N_counts_batch[:, j] = count_j
            
            # Save for stats
            N_samples[j].extend(count_j.cpu().tolist())

        # --- 3. Compute K_i (Derived Counts) ---
        # Formula: K_i = Sum_{j>=i} binom(j, i) * N_j
        # We can do this via matrix multiplication for the whole batch
        # N_counts_batch: (B, j)
        # binom_coeffs: (j, i) -> We want (j, i) where i are cols
        
        # We use the precomputed coeffs. Move to device.
        coeffs = binom_coeffs.to(x_batch.device)
        
        # Matmul: (B, j) x (j, i) -> (B, i)
        # resulting K_values shape: (B, effective_k + 1)
        # Index 0 will be K_0 (which is Sum N_j = Total History), usually uninteresting but calc anyway
        K_values_batch = torch.matmul(N_counts_batch, coeffs)
        
        for i in range(effective_k + 1):
            K_samples[i].extend(K_values_batch[:, i].cpu().tolist())
            
        total_samples += B
        
        # --- 4. Convergence Check ---
        if total_samples >= min_samples:
            max_sem = 0.0
            # Check convergence on K variables (usually the ones we care about)
            for i in range(effective_k + 1):
                data = np.array(K_samples[i])
                if len(data) < 2: continue
                sem = np.std(data, ddof=1) / np.sqrt(len(data))
                if sem > max_sem: max_sem = sem
            
            if max_sem < variance_threshold:
                print(f"Converged at {total_samples} samples.")
                break

    # --- 5. Compile Results ---
    results = {'N': {}, 'K': {}}
    
    for j, data in N_samples.items():
        arr = np.array(data)
        results['N'][j] = {'mean': np.mean(arr), 'var': np.var(arr, ddof=1)}
        
    for i, data in K_samples.items():
        arr = np.array(data)
        results['K'][i] = {'mean': np.mean(arr), 'var': np.var(arr, ddof=1)}
        
    return results


def evaluate_model(model, x_val, y_val):
    """
    Evaluates the model on a fixed validation set.
    """
    model.eval()  # Switch to eval mode
    with torch.no_grad():
        probs = model(x_val)
        # Numerical stability handling
        loss = F.nll_loss(torch.log(probs.clamp(min=1e-9)), y_val)
    model.train()  # Switch back to train mode
    return loss.item()


def approximate_beta(order_k, vocab_size, seq_len, alpha) -> float:
    return np.log(
        1
        + (
            vocab_size
            / (
                (
                    (
                        1.0
                        + (alpha * (vocab_size ** (order_k + 1))) / (seq_len - order_k)
                    )
                    ** (1.0 / order_k)
                )
                - 1
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Constrained Attention on Markov Chains"
    )

    # Model & Data Params
    parser.add_argument("--vocab_size", type=int, default=5, help="Size of vocabulary")
    parser.add_argument(
        "--seq_len", type=int, default=50, help="Length of context sequence"
    )
    parser.add_argument(
        "--order_k", type=int, default=1, help="Order of the history window (k)"
    )
    parser.add_argument(
        "--model_k", type=int, default=0, help="Order of the model"
    )
    parser.add_argument(
        "--mask_first",
        action="store_true",
        help="Whether to mask the first token in attention",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Dirichlet concentration parameter"
    )
    parser.add_argument(
        "--init_scale", type=float, default=0.0, help="Initialization scale"
    )

    # Training Params
    parser.add_argument(
        "--couple_diagonals",
        action="store_true",
        help="Force parameters to be shared along diagonals (Toeplitz structure)",
    )
    parser.add_argument(
        "--train_sub_pattern",
        action="store_true",
        help="Only train sub pattern (gradient masking), keep others at 0",
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--steps", type=int, default=50000, help="Number of training steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Steps between printing training loss",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Steps between running validation",
    )
    parser.add_argument(
        "--val_size", type=int, default=5000, help="Fixed validation set size"
    )
    parser.add_argument(
        "--estimate_counts", action="store_true", help="Estimate the counts"
    )

    # WandB Params
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="induction-head-markov",
        help="WandB Project Name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="WandB Entity (User/Team) Name"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, config=vars(args)
        )

    # Setup
    if args.model_k == 0:
        model_k = args.order_k + 1
    else:
        model_k = args.model_k

    # --- Generate Fixed Validation Set ---
    print(f"Generating fixed validation set (size {args.val_size})...")
    x_val, y_val = generate_dirichlet_markov_data(
        args.val_size, args.seq_len, args.vocab_size, args.order_k, args.alpha
    )

    if args.estimate_counts:
        print(f"Estimating statistics (mask_first={args.mask_first}, context_len={model_k-1})...")
        stats = estimate_overlap_distribution(
            generate_dirichlet_markov_data,
            args.batch_size,
            args.seq_len,
            args.vocab_size,
            args.order_k,
            args.alpha,
            max_k=model_k, # This is the model order (e.g. 2 implies 1 context token)
            mask_first=args.mask_first,
            variance_threshold=0.05 # Relaxed threshold as Variance of K can be huge
        )
        
        print("\n=== N_j Statistics (Exact Overlaps) ===")
        print(f"{'j':<5} | {'Mean':<10} | {'Variance':<10}")
        print("-" * 30)
        for j, v in stats['N'].items():
            print(f"{j:<5} | {v['mean']:<10.4f} | {v['var']:<10.4f}")

        print("\n=== K_i Statistics (Sub-pattern Matches) ===")
        print(f"{'i':<5} | {'Mean':<10} | {'Variance':<10}")
        print("-" * 30)
        for i, v in stats['K'].items():
            print(f"{i:<5} | {v['mean']:<10.4f} | {v['var']:<10.4f}")

        print("\n=== Random Statistics (Sub-pattern Matches) ===")
        print(f"{'i':<5} | {'Estimate':<10}")
        print("-" * 30)
        for i, v in stats['K'].items():
            print(f"{i:<5} | {stats['K'][0]['mean'] * ((args.vocab_size) ** -i) * scipy.special.comb(model_k-1, i)}")

        return 0

    # --- COMPUTE OPTIMAL LOSS (BASELINE) ---
    print("Computing Bayes Optimal Loss for Validation Set...")
    # This is constant for the fixed validation set
    optimal_val_loss = compute_optimal_loss(
        x_val, y_val, args.vocab_size, args.order_k, args.alpha
    )
    print(f"Optimal Val Loss (Bayes Limit): {optimal_val_loss:.4f}")

    model = ConstrainedAttentionModel(
        args.vocab_size,
        init_scale=args.init_scale,
        order_k=model_k,
        mask_first=args.mask_first,
        couple_diagonals=args.couple_diagonals,  # New argument
    )

    # === GRADIENT MASKING LOGIC ===
    if args.couple_diagonals:
        print(f"Configuration: Coupled Diagonals Mode (Toeplitz).")
        if args.train_sub_pattern:
            print(
                "  -> AND 'train_sub_pattern' active: Masking 1D parameters to train ONLY offset +1."
            )
            # Calculate index for offset +1
            # Center (offset 0) is at index K-1. Offset +1 is at K.
            target_idx = model_k

            mask = torch.zeros_like(model.diag_params)
            if target_idx < mask.shape[0]:
                mask[target_idx] = 1.0

            # Register hook on the 1D parameter vector
            model.diag_params.register_hook(lambda grad: grad * mask.to(grad.device))
    else:
        if args.train_sub_pattern:
            print(
                f"Configuration: Matrix Mode. Training ONLY induction diagonals (offset +1)."
            )
            mask = torch.zeros(model_k, model_k)
            for i in range(model_k - 1):
                mask[i, i + 1] = 1.0
            model.params.register_hook(lambda grad: grad * mask.to(grad.device))
        else:
            print(f"Configuration: Full Matrix Mode.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    approximated_beta = approximate_beta(
        args.order_k, args.vocab_size, args.seq_len, args.alpha
    )
    print(f"Approximated beta {approximated_beta}")
    if not args.no_wandb:
        wandb.run.summary["approximated_beta"] = approximated_beta

    # Training Loop
    for step in range(args.steps):
        # Generate fresh batch
        x_batch, y_batch = generate_dirichlet_markov_data(
            args.batch_size, args.seq_len, args.vocab_size, args.order_k, args.alpha
        )

        optimizer.zero_grad()
        probs = model(x_batch)
        loss = F.nll_loss(torch.log(probs.clamp(min=1e-9)), y_batch)
        loss.backward()
        optimizer.step()

        # Logging
        if step % args.log_interval == 0 or step % args.eval_interval == 0:
            # --- 1. Prepare Data ---
            with torch.no_grad():
                matrix_data = model.get_effective_params().cpu().numpy()
            matrix_rounded = np.around(matrix_data, 3)

            # --- 2. Create Plot using Matplotlib ---
            # This is robust and works on all WandB versions
            fig, ax = plt.subplots(figsize=(5, 5))
            cax = ax.matshow(matrix_rounded, cmap="viridis")

            for i in range(model_k):
                for j in range(model_k):
                    ax.text(
                        j,
                        i,
                        str(matrix_rounded[i, j]),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

            labels = [f"L{i}" for i in range(model_k)]
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
                "parameter_matrix": wandb.Image(fig),
            }

            # Close plot to prevent memory leaks
            plt.close(fig)

            for i in range(model_k):
                for j in range(model_k):
                    val = matrix_data[i, j].item()
                    # Key format: "C_q{row}_k{col}"
                    log_dict[f"param/C_q{i}_k{j}"] = val

            # Print a quick summary to the console (just the induction diagonal)
            # We assume the "Induction Head" logic lies on the +1 diagonal (j = i + 1)
            induction_vals = []
            for i in range(model_k-1):
                induction_vals.append(f"{matrix_data[i, i+1].item():.2f}")

            print(
                f"Step {step}: Loss = {loss.item():.4f} | Induction Diags: {induction_vals}"
            )

            if step % args.eval_interval == 0:
                val_loss = evaluate_model(model, x_val, y_val)
                print(f"    >>> EVAL: Val Loss = {val_loss:.4f}")
                log_dict.update(
                    {
                        "val_loss": val_loss,
                        "excess_loss": val_loss - optimal_val_loss,
                        "optimal_loss": optimal_val_loss,
                    }
                )

            if not args.no_wandb:
                wandb.log(log_dict)

    # Final Result
    print("-" * 30)
    print("Optimization Complete.")
    print(f"Final Parameters:\n{model.get_effective_params().data}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
