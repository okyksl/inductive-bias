import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import wandb

class ConstrainedAttentionModel(nn.Module):
    def __init__(self, vocab_size, mask_first: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_first = mask_first
        
        # The 4 parameters determining the scale of the identities
        # Initialized with small random noise
        self.params = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        """
        x: (Batch, T) - Sequence of token indices
        """
        B, T = x.shape
        
        # 1. Embeddings (One-Hot)
        # Shape: (B, T, V)
        emb = F.one_hot(x, num_classes=self.vocab_size).float()
        
        # 2. Layer 1: Copying [x_t, x_{t-1}]
        # Create context pairs
        x_curr = emb
        x_prev = torch.cat([torch.zeros(B, 1, self.vocab_size, device=x.device), emb[:, :-1, :]], dim=1)
        
        # 3. Layer 2: Attention Mechanism
        # Query: Last position (T)
        q_curr = x_curr[:, -1:, :] # x_T
        q_prev = x_prev[:, -1:, :] # x_{T-1}
        
        # Remove the first index (no attention to first) if requested
        if self.mask_first:
            x_curr_k = x_curr[:, 1:, :]
            x_prev_k = x_prev[:, 1:, :]
        else:
            x_curr_k = x_curr
            x_prev_k = x_prev

        # Keys: All positions
        k_curr = x_curr_k
        k_prev = x_prev_k
        
        # Calculate Scores (Batch, 1, T)
        # Note: If mask_first is True, T changes for keys. 
        # But q stays (B, 1, V). Transpose ensures correct dim match.
        score_1 = torch.matmul(q_curr, k_curr.transpose(1, 2)) # x_T . x_t
        score_2 = torch.matmul(q_curr, k_prev.transpose(1, 2)) # x_T . x_{t-1} (Induction)
        score_3 = torch.matmul(q_prev, k_curr.transpose(1, 2)) # x_{T-1} . x_t
        score_4 = torch.matmul(q_prev, k_prev.transpose(1, 2)) # x_{T-1} . x_{t-1}
        
        # Linear combination of identity matrices
        scores = (self.params[0] * score_1 + 
                  self.params[1] * score_2 + 
                  self.params[2] * score_3 + 
                  self.params[3] * score_4)
        
        # Causal Masking (Mask out the query position T to force looking at history)
        # If we masked first, the sequence length of keys is T-1 or similar, 
        # we need to be careful with indices. 
        # For simplicity in this specific script context where we predict last token:
        # We just want to ensure we don't attend to the *very last* token itself which is the query.
        
        mask = torch.ones_like(scores)
        # Mask the last position (which corresponds to the query token itself)
        mask[:, :, -1] = 0 
        scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Attention Weights (Normalized counts)
        attn_weights = F.softmax(scores, dim=-1) # (B, 1, T)
        
        # 5. Value Aggregation
        values = x_curr_k
        # (B, 1, T) x (B, T, V) -> (B, V)
        output = torch.matmul(attn_weights, values).squeeze(1)
        
        # Return raw probabilities directly
        return output

def generate_dirichlet_markov_data(batch_size, seq_len, vocab_size, alpha):
    """
    Generates sequences where transition probabilities are sampled from Dirichlet(alpha).
    """
    data = []
    targets = []
    
    for _ in range(batch_size):
        # 1. Sample transition matrix from Dirichlet prior
        transitions = torch.distributions.Dirichlet(torch.full((vocab_size,), alpha)).sample((vocab_size,))
        
        # 2. Generate Sequence
        seq = [torch.randint(0, vocab_size, (1,)).item()]
        for _ in range(seq_len):
            curr = seq[-1]
            probs = transitions[curr]
            next_token = torch.multinomial(probs, 1).item()
            seq.append(next_token)
            
        data.append(torch.tensor(seq[:-1])) # x_1 ... x_T
        targets.append(torch.tensor(seq[-1])) # x_{T+1}
        
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
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--val_size", type=int, default=5000, help="Fixed validation set size")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration parameter")
    parser.add_argument("--mask_first", action="store_true", help="Whether to mask the first token in attention")
    
    # Training Params
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_sub_pattern", action="store_true", help="Only train sub pattern, keep others at 0")
    parser.add_argument("--log_interval", type=int, default=50, help="Steps between printing training loss")
    parser.add_argument("--eval_interval", type=int, default=500, help="Steps between running validation")

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
    model = ConstrainedAttentionModel(args.vocab_size, mask_first=args.mask_first)
    if args.train_sub_pattern:
        print("Configuration: Training ONLY c2. c1, c3, c4 will remain 0.0.")
        # Create a mask where only index 1 (c2) is 1.0, others are 0.0
        mask = torch.tensor([0., 1., 0., 0.])
        
        # Register a hook that multiplies the gradient by this mask
        # This happens right after loss.backward() and before optimizer.step()
        model.params.register_hook(lambda grad: grad * mask.to(grad.device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"Initial Params: {model.params.data}")

    # --- Generate Fixed Validation Set ---
    print(f"Generating fixed validation set (size {args.val_size})...")
    x_val, y_val = generate_dirichlet_markov_data(
        args.val_size, args.seq_len, args.vocab_size, args.alpha
    )

    # Training Loop
    for step in range(args.steps):
        # Generate fresh batch
        x_batch, y_batch = generate_dirichlet_markov_data(
            args.batch_size, args.seq_len, args.vocab_size, args.alpha
        )
        
        optimizer.zero_grad()
        probs = model(x_batch)
        loss = F.nll_loss(
            torch.log(probs.clamp(min=1e-9)), 
            y_batch
        )
        loss.backward()
        optimizer.step()
        
        c1, c2, c3, c4 = model.params.tolist()
        log_dict = {}

        # 1. Frequent Logging (Training Loss)
        if step % args.log_interval == 0:
            print(f"Step {step}: Train Loss = {loss.item():.4f} | c2={c2:.4f}")
            log_dict.update({
                "train_loss": loss.item(),
                "c1": c1, "c2": c2, "c3": c3, "c4": c4,
                "step": step
            })

        # 2. Infrequent Evaluation (Validation Loss)
        if step % args.eval_interval == 0:
            val_loss = evaluate_model(model, x_val, y_val)
            print(f"    >>> EVAL: Val Loss = {val_loss:.4f}")
            log_dict.update({"val_loss": val_loss})

        # Send to WandB if we have anything to log
        if log_dict and not args.no_wandb:
            wandb.log(log_dict)

    # Final Result
    print("-" * 30)
    print("Optimization Complete.")
    print(f"Final Parameters: {model.params.data}")
    
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()