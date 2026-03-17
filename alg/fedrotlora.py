import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--lam', type=float, default=0.5,
                        help="Soft rotation strength lambda in [0, 1]. "
                             "0 = no alignment (naive), 1 = hard Procrustes alignment.")
    return parser.parse_args()


def _procrustes(M):
    """Closed-form solution to the orthogonal Procrustes problem.

    Given M = U Sigma V^T, returns R* = V diag(1,...,det(UV^T)) U^T.
    This satisfies R^T R = I and det(R) = +1 (SO(r)).
    """
    U, _, Vh = torch.linalg.svd(M)
    V = Vh.mH  # conjugate transpose = transpose for real matrices
    sign = torch.det(U @ Vh)
    d = torch.ones(M.shape[0], device=M.device, dtype=M.dtype)
    d[-1] = sign
    return V @ torch.diag(d) @ U.mH


def _soft_rotation(R_star, lam):
    """Interpolate between identity and R_star, then project back onto SO(r).

    R' = (1 - lam) * I + lam * R_star
    R_soft = nearest SO(r) matrix to R'
    """
    I = torch.eye(R_star.shape[0], device=R_star.device, dtype=R_star.dtype)
    R_prime = (1.0 - lam) * I + lam * R_star
    U, _, Vh = torch.linalg.svd(R_prime)
    sign = torch.det(U @ Vh)
    d = torch.ones(R_star.shape[0], device=R_star.device, dtype=R_star.dtype)
    d[-1] = sign
    return U @ torch.diag(d) @ Vh


def _align_factors(A_i, B_i, A_ref, B_ref, t, lam):
    """Apply rotational alignment to a single LoRA layer pair.

    A_i:   [r, d]  local lora_A after training
    B_i:   [d, r]  local lora_B after training
    A_ref: [r, d]  global reference lora_A from previous round
    B_ref: [d, r]  global reference lora_B from previous round
    t:     int     current round index
    lam:   float   soft rotation strength

    Returns (A_tilde, B_tilde) with the same shapes as inputs.
    The semantic update is preserved: B_tilde @ A_tilde == B_i @ A_i.
    """
    orig_dtype = A_i.dtype
    # Work in float32 for numerical stability
    A_i   = A_i.float()
    B_i   = B_i.float()
    A_ref = A_ref.float()
    B_ref = B_ref.float()

    if t % 2 == 1:
        # A-alignment round: min_{R} ||(R)^T A_i - A_ref||^2_F
        # Correlation matrix M = A_ref @ A_i^T  (r×r)
        M = A_ref @ A_i.T
    else:
        # B-alignment round: min_{R} ||B_i R - B_ref||^2_F
        # Correlation matrix M = B_ref^T @ B_i  (r×r)
        M = B_ref.T @ B_i

    R_star = _procrustes(M)
    R_soft = _soft_rotation(R_star, lam)

    # A_tilde = R_soft^T @ A_i,  B_tilde = B_i @ R_soft
    # Correctness: B_tilde @ A_tilde = B_i @ R_soft @ R_soft^T @ A_i = B_i @ A_i ✓
    A_tilde = R_soft.T @ A_i
    B_tilde = B_i @ R_soft

    return A_tilde.to(orig_dtype), B_tilde.to(orig_dtype)


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        # Cache the current global reference (previous round's aggregated adapter)
        # before local training overwrites the model weights.
        A_ref = {k: v.clone() for k, v in self.server.global_lora.items() if "lora_A" in k}
        B_ref = {k: v.clone() for k, v in self.server.global_lora.items() if "lora_B" in k}

        # --- Local training ---
        self.trainer.train(model)

        # Collect locally trained LoRA factors
        local_lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

        t = self.server.round
        lam = self.server.args.lam

        # Skip alignment in round 0: LoRA is initialised with B0 A0 = 0,
        # so the reference is degenerate and cannot define a meaningful subspace.
        if t == 0:
            self.lora = local_lora
            return

        aligned = {}
        for a_key in [k for k in local_lora if "lora_A" in k]:
            b_key = a_key.replace("lora_A", "lora_B")
            A_tilde, B_tilde = _align_factors(
                A_i=local_lora[a_key],
                B_i=local_lora[b_key],
                A_ref=A_ref[a_key],
                B_ref=B_ref[b_key],
                t=t,
                lam=lam,
            )
            aligned[a_key] = A_tilde
            aligned[b_key] = B_tilde

        self.lora = aligned


class Server(FTBaseServer):
    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()
