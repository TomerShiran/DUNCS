import math
import torch

def _vecF_4d_to_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Column-major vectorization on the last two dims:
      x: (B, U, U, K) -> (B, U*U, K)
    We transpose the U-dims then row-major reshape, which equals Fortran vec.
    """
    B, U, U2, K = x.shape
    assert U == U2, "expected (B,U,U,K)"
    return x.transpose(1, 2).reshape(B, U * U, K)


def _broadcast_theta_snr_L(thetas_rad, snr_db, L, device):
    """
    Make thetas -> (B,K), snr_db -> (B,K), L -> (B,)
    Accepts scalars, (K,), (B,), or (B,K).
    """
    thetas = torch.as_tensor(thetas_rad, dtype=torch.float64, device=device)
    if thetas.dim() == 1:
        thetas = thetas.unsqueeze(0)
    B_theta, K = thetas.shape

    snr = torch.as_tensor(snr_db, dtype=torch.float64, device=device)
    if   snr.dim() == 0:
        B_snr = 1
    elif snr.dim() == 1:
        B_snr = snr.shape[0]
    elif snr.dim() == 2:
        B_snr = snr.shape[0]
        if snr.shape[1] != K:
            raise ValueError(f"snr_db has K={snr.shape[1]} but thetas has K={K}")
    else:
        raise ValueError("snr_db must be scalar, (B,), or (B,K)")

    L_t = torch.as_tensor(L, dtype=torch.float64, device=device)
    if   L_t.dim() == 0: B_L = 1
    elif L_t.dim() == 1: B_L = L_t.shape[0]
    else: raise ValueError("L_snapshots must be scalar or (B,)")

    B = int(max(B_theta, B_snr, B_L))

    # tile/broadcast
    if B_theta == 1 and B > 1:
        thetas = thetas.expand(B, K)
    elif B_theta not in (1, B):
        raise ValueError(f"thetas batch {B_theta} != {B}")

    if snr.dim() == 0:
        snr = snr.expand(B, K)
    elif snr.dim() == 1:
        if B_snr == 1 and B > 1:
            snr = snr.expand(B)
            B_snr = B
        if B_snr != B:
            raise ValueError(f"snr_db batch {B_snr} != {B}")
        snr = snr.view(B, 1).expand(B, K)
    else:  # (B_snr, K)
        if B_snr == 1 and B > 1:
            snr = snr.expand(B, K)
            B_snr = B
        if snr.shape != (B, K):
            raise ValueError(f"snr_db shape {tuple(snr.shape)} != ({B},{K})")

    if L_t.dim() == 0:
        L_t = L_t.expand(B)
    elif L_t.dim() == 1:
        if L_t.shape[0] == 1 and B > 1:
            L_t = L_t.expand(B)
        if L_t.shape[0] != B:
            raise ValueError(f"L length {L_t.shape[0]} != B={B}")

    return thetas, snr, L_t  # (B,K), (B,K), (B,)


def calculate_sncr_crb(
    sparse_array: torch.LongTensor,
    thetas_rad: torch.Tensor,      # (B,K) or (K,)   -- radians
    snr_db: float | torch.Tensor,  # scalar or (B,) or (B,K)   -- per-source SNR in dB
    L_snapshots: int | float | torch.Tensor,
    d: float = 0.5,
    sigma2: float = 1.0,
    return_per_angle: bool = True,
):
    """
    SNCR-CRB as derived in "Structured Nyquist Correlation Reconstruction for
                            DOA Estimation With Sparse Arrays"

      F = L * [vec(∂Ryy/∂β)]^H * ( S^T ⊗ S ) * [vec(∂Ryy/∂β)]
      S = Φ^T ( Φ Ryy Φ^H )^{-1} Φ,   Ryy = A diag(p) A^H + σ^2 I

    β = [θ_1..θ_K, P_1..P_K, σ^2]^T, with θ in radians.
    Returns per-angle variances (rad^2) or RMSE (rad).

    Notes:
      * All large Kroneckers are formed explicitly, as in the paper.
      * Inversion of (Φ Ryy Φ^H) is done via Cholesky solve.
      * dtype is hard-coded to complex128, real parts are kept where theory dictates.
    """
    device = sparse_array.device if isinstance(sparse_array, torch.Tensor) else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype_c = torch.complex128
    Sidx = torch.as_tensor(sparse_array, dtype=torch.long, device=device)
    M = Sidx.numel()
    U = int(Sidx.max().item() + 1)  # presumed contiguous ULA grid 0..max(S)

    Phi = torch.zeros(M, U, dtype=torch.float64, device=device)
    Phi[torch.arange(M, device=device), Sidx] = 1.0
    Phi_c = Phi.to(dtype_c)

    # --- broadcast inputs ---
    thetas, snr_db_t, L_t = _broadcast_theta_snr_L(thetas_rad, snr_db, L_snapshots, device)
    B, K = thetas.shape

    # --- ULA steering and derivatives on U sensors ---
    u = torch.arange(U, dtype=torch.float64, device=device)  # 0..U-1
    phase_const = 2.0 * math.pi * d
    sin_t = torch.sin(thetas)          # (B,K)
    cos_t = torch.cos(thetas)

    phase = -phase_const * u.view(1, U, 1) * sin_t.view(B, 1, K)  # (B,U,K)
    A = torch.exp(1j * phase).to(dtype_c)                         # (B,U,K)
    dA = ((-1j * phase_const) * u.view(1, U, 1) * cos_t.view(B, 1, K)).to(dtype_c) * A

    # --- stochastic powers from SNR: p_k = (SNR_lin)*σ^2
    snr_lin = 10.0 ** (snr_db_t / 10.0)         # (B,K)
    p = (snr_lin * float(sigma2)).to(torch.float64)  # (B,K)
    p_c = p.to(dtype_c)

    # --- virtual covariance Ryy and measured Rxx ---
    Ap = A * p_c.view(B, 1, K)
    Iu = torch.eye(U, dtype=dtype_c, device=device)
    Ryy = Ap @ A.conj().transpose(-2, -1) + float(sigma2) * Iu              # (B,U,U)
    Rxx = torch.einsum('mu, buv, nv -> bmn', Phi_c, Ryy, Phi_c.conj())      # (B,M,M)

    # --- S = Φ^T (Φ Ryy Φ^H)^{-1} Φ  via Cholesky solve (exact inverse, numerically stable) ---
    S_list = []
    for b in range(B):
        Lc = torch.linalg.cholesky(Rxx[b])                # Rxx = L L^H
        Rinv = torch.cholesky_inverse(Lc)                 # (ΦRyyΦ^H)^{-1}
        S_b = Phi_c.T @ Rinv @ Phi_c                      # (U×U)
        S_list.append(S_b)
    S_eff = torch.stack(S_list, dim=0).contiguous()       # (B,U,U)

    # --- Jacobian in ULA domain: J_y = [ D_theta | D_p | d_sigma ]  (size U^2 × (2K+1)) ---
    outer_aa   = torch.einsum('buk,bvk->buvk', A,  A.conj())    # (B,U,U,K)
    outer_da_a = torch.einsum('buk,bvk->buvk', dA, A.conj())
    outer_a_da = torch.einsum('buk,bvk->buvk', A,  dA.conj())

    D_theta4d = (outer_da_a + outer_a_da) * p_c.view(B, 1, 1, K)  # (B,U,U,K)
    D_p4      = outer_aa                                          # (B,U,U,K)
    d_sigma4  = Iu.view(1, U, U, 1).expand(B, U, U, 1)            # (B,U,U,1)

    D_theta = _vecF_4d_to_2d(D_theta4d)   # (B,U^2,K)
    D_p     = _vecF_4d_to_2d(D_p4)        # (B,U^2,K)
    d_sigma = _vecF_4d_to_2d(d_sigma4)    # (B,U^2,1)

    J_y = torch.cat([D_theta, D_p, d_sigma], dim=2).to(dtype_c)   # (B,U^2, 2K+1)
    P_tot = 2 * K + 1

    # --- Build Kron(S^T, S) and assemble F exactly as Eq. (33) ---
    F_list = []
    for b in range(B):
        S_b = S_eff[b]
        # (S^T ⊗ S)  -- use plain transpose (no conjugation) to match the equation
        S_kron = torch.kron(S_b.transpose(-2, -1).contiguous(), S_b.contiguous())  # (U^2, U^2)
        Jy = J_y[b]                                     # (U^2, P_tot)
        F_b = (Jy.conj().transpose(0, 1) @ (S_kron @ Jy)).real  # (P_tot, P_tot), real Hermitian
        # Symmetrize numerically and scale by L
        F_b = 0.5 * (F_b + F_b.transpose(0, 1))
        F_b = L_t[b].item() * F_b
        F_list.append(F_b)
    F = torch.stack(F_list, dim=0)                      # (B, P_tot, P_tot), real

    # --- Invert F exactly (no floors); fallback to pinv if singular ---
    CRB_list = []
    eye_cache = torch.eye(P_tot, dtype=torch.float64, device=device)
    for b in range(B):
        Fb = F[b]
        try:
            CRB_b = torch.linalg.solve(Fb, eye_cache)  # exact inverse if nonsingular
        except torch.linalg.LinAlgError:
            raise Exception("FIM is singular")
            # CRB_b = torch.linalg.pinv(Fb)              # Moore–Penrose if needed
        CRB_list.append(CRB_b)
    CRB = torch.stack(CRB_list, dim=0)                 # (B, P_tot, P_tot)

    # θ-block variances
    CRB_theta = CRB[:, :K, :K]
    per_angle_var = torch.diagonal(CRB_theta, dim1=-2, dim2=-1).clamp_min(0.0)  # (B,K)

    if return_per_angle:
        return per_angle_var  # rad^2
    # RMSE in radians (as in the paper’s definition, averaged over sources)
    return torch.sqrt(per_angle_var.mean(dim=1))

