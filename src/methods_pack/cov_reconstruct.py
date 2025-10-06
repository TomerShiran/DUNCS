from abc import ABC, abstractmethod
import torch
import cvxpy as cp
import numpy as np

from src.utils import build_phi, psd_proj, toeplitz_proj, hermitian_proj, svt, device
from src.system_model import SystemModel


class CovReconstructor(ABC):
    """
    Given observation tensor x from S sensors (B×S×T),
    returns batch of completed coarray covariances (B×U×U).
    """
    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SampleCov(CovReconstructor):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return sample_covariance(x).to(device=device)


class SpatialSmoothingReconstructor(CovReconstructor):
    """
    Calculates the covariance matrix using forward–backward spatial smoothing technique.
    """
    def __init__(self, sub_array_size=None):
        self.sub_array_size = sub_array_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure x has three dimensions (batch, sensors, samples)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, sensor_number, samples_number = x.shape

        # Define subarray size and the number of overlapping subarrays
        if self.sub_array_size is None:
            self.sub_array_size = sensor_number // 2 + 1

        number_of_sub_arrays = sensor_number - self.sub_array_size + 1

        # Initialize the smoothed covariance matrix
        Rx_smoothed = torch.zeros(batch_size, self.sub_array_size, self.sub_array_size,
                                  dtype=torch.complex128, device=x.device)

        for j in range(number_of_sub_arrays):
            # Extract the j-th subarray
            x_sub = x[:, j:j + self.sub_array_size, :]

            # Forward covariance calculation
            cov_forward = torch.einsum("bmt, btl -> bml", x_sub,
                                       torch.conj(x_sub).transpose(1, 2)) / (samples_number - 1)

            # backward processing: take the complex conjugate before flipping
            x_sub_back = torch.flip(torch.conj(x_sub), dims=[1])
            cov_backward = torch.einsum("bmt, btl -> bml", x_sub_back,
                                        torch.conj(x_sub_back).transpose(1, 2)) / (samples_number - 1)

            # Average the forward and backward covariances for this subarray
            cov_fb = 0.5 * (cov_forward + cov_backward)

            # Aggregate over all subarrays
            Rx_smoothed += cov_fb / number_of_sub_arrays

        return Rx_smoothed


class AveragingReconstructor(CovReconstructor):
    """
    Calculates the virtual array covariance matrix, based on the paper: "Remarks on the Spatial Smoothing Step in
    Coarray MUSIC"

     Parameters
     ----------
      X (torch.Tensor): Input samples matrix.
      system_model (SystemModel): settings of the system model

    Returns
    -------
    Rx (torch.Tensor): virtual array's covariance matrix
    """
    def __init__(self, sys_model: SystemModel):
        self.L = len(sys_model.virtual_array_ula_seg)
        self.virtual_array = sys_model.virtual_array_ula_seg
        self.diff_array = sys_model.array[:, None] - sys_model.array[None, :]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        R_real_array = sample_covariance(x)

        Rx = torch.zeros(R_real_array.shape[0], self.L, self.L, dtype=torch.complex128)
        x_s_diff = torch.zeros(x.shape[0], 2 * self.L - 1, dtype=torch.complex128)  # x.shape[0] = batch size
        max_sensor = np.max(self.virtual_array)

        for i, lag in enumerate(range(-max_sensor, max_sensor + 1)):
            pairs = torch.from_numpy(self.diff_array) == lag
            if pairs.any():
                x_s_diff[:, i] = torch.mean(R_real_array[:, pairs], dim=1)

        for j in range(self.L):
            start_idx = self.L - 1 - j
            Rx[:, :, j] = x_s_diff[:, start_idx:start_idx + self.L]

        return Rx.to(device=device)


class ADMMReconstructor(CovReconstructor):
    """
    Batched ADMM solver for the nuclear-norm covariance completion

    min ‖ΦRΦᴴ − Rₓₓ‖_F² + μ‖R‖_*
    s.t.  R  Hermitian–Toeplitz – PSD.

    ADMM implementation to the method proposed at:
    'Structured Nyquist Correlation Reconstruction for DOA Estimation With Sparse Arrays, 2023'
    """
    def __init__(self, sys_model: SystemModel, mu: float = 2.5e-3,
                     rho: float = 2, max_iter=400, tol_primal: float = 1e-7,
                     tol_dual: float = 1e-7,
                     verbose: bool = False):
        self.sys = sys_model
        self.phi = build_phi(self.sys.array)  # (|S|,|U|)
        self.phi_H = self.phi.t()
        self.U = self.phi.shape[1]  # |U|

        # mask diagonal  (P = Φᴴ Φ) → 1-D of length |U|²
        m = (self.phi_H @ self.phi).diag()  # (|U|,)
        self.P = (m[:, None] * m[None, :]).flatten()  # (|U|²,)

        self.mu = mu
        self.rho = rho
        self.max_iter = max_iter
        self.tol_primal = tol_primal
        self.tol_dual = tol_dual
        self._verbose = verbose

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        R_tilde : (B,|U|,|U|) complex tensor
        """
        Rx = sample_covariance(x)
        B, S, _ = Rx.shape
        U = self.U
        dev, dtype = Rx.device, Rx.dtype

        phi = self.phi.to(dev, dtype)
        phi_H = self.phi_H.to(dev, dtype)

        # ------------------------------------------------------------
        # measured part  Φᴴ Rₓₓ Φ  →  (B,|U|,|U|)
        # ------------------------------------------------------------
        meas = phi_H @ Rx @ phi
        vec_meas = meas.reshape(B, -1)  # (B, |U|²)

        # ------------------------------------------------------------
        # diagonal coefficients  (mask + 2ρI)⁻¹  (1-D then broadcast)
        # ------------------------------------------------------------
        inv_coeff = 1.0 / (self.P.to(dev, dtype) + 2 * self.rho)
        inv_coeff = inv_coeff.expand(B, -1)  # (B, |U|²)

        # ------------------------------------------------------------
        # initial variables  (B,|U|,|U|)
        # ------------------------------------------------------------
        R = hermitian_proj(meas)
        S = R.clone()
        T = toeplitz_proj(R)
        Udual = torch.zeros_like(R)
        Vdual = torch.zeros_like(R)

        S_prev, T_prev = S.clone(), T.clone()

        # ------------------------------------------------------------
        # ADMM iterations
        # ------------------------------------------------------------

        for k in range(self.max_iter):
            # R-update  (diagonal solve, batched)
            rhs = vec_meas + self.rho * (S - Udual + T - Vdual).reshape(B, -1)
            vec_R = inv_coeff * rhs
            R = vec_R.view(B, U, U)

            # S-update  (SVT)
            Z = R + Udual
            S = svt(Z, self.mu / self.rho)

            # T-update  (Herm-Toeplitz-PSD)
            W = R + Vdual
            T = psd_proj(toeplitz_proj(hermitian_proj(W)))

            # dual ascent
            Udual += R - S
            Vdual += R - T

            # convergence criteria (batch max)
            r_norm = torch.max(
                (R - S).flatten(1).norm(dim=1),
                (R - T).flatten(1).norm(dim=1)
            ).max()  # global primal residual

            s_norm = self.rho * torch.max(
                (S - S_prev).flatten(1).norm(dim=1),
                (T - T_prev).flatten(1).norm(dim=1)
            ).max()  # global dual residual

            if self._verbose and (k % 25 == 0):
                print(f"iter {k:4d} | primal {r_norm:.3e} | dual {s_norm:.3e}")

            if r_norm < self.tol_primal and s_norm < self.tol_dual:
                break

            S_prev.copy_(S)
            T_prev.copy_(T)

        return T


class ADMMReconstructorCVXPY(CovReconstructor):
    """
    Solver for the nuclear-norm covariance completion using CVXPY package

    min ‖ΦRΦᴴ − Rₓₓ‖_F² + μ‖R‖_*
    s.t.  R  Hermitian–Toeplitz – PSD.

    ADMM implementation to the method proposed at:
    'Structured Nyquist Correlation Reconstruction for DOA Estimation With Sparse Arrays, 2023'
    """
    def __init__(self, sys_model: SystemModel, mu: float = 2.5e-3, **unused_kwargs):
        self.sys = sys_model
        self.phi = build_phi(self.sys.array)  # (|S|,|U|)
        self.phi_H = self.phi.t()
        self.U = self.phi.shape[1]  # |U|
        self.S = self.phi.shape[0]  # |S|

        self.mu = mu
        self._solver = "SCS"   # Splitting Conic Solver, It solves primal-dual problems

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: observations Tensor
        mu : float
            nuclear-norm weight μ in (17)
        solver : str
            CVXPY solver name (“SCS”, “MOSEK”, "CVXOPT", ...)

        Returns
        -------
        R_tilde : torch.Tensor (|U|, |U|)
            Toeplitz-PSD covariance fitted by nuclear-norm minimisation
        """
        R_xx = sample_covariance(x).cpu().numpy()  # (|S|,|S|)

        # CVXPY variable (complex Hermitian)
        R = cp.Variable((self.U, self.U), hermitian=True)

        # Toeplitz constraint: equality of diagonals
        toeplitz_constraints = []
        for k in range(-self.U + 1, self.U):
            diag = cp.diag(R, k)
            toeplitz_constraints.append(diag == cp.mean(diag))

        # PSD constraint
        constraints = toeplitz_constraints + [R >> 0]

        # Objective
        fit = cp.norm(self.phi @ R @ self.phi_H - R_xx, p='fro')
        obj = cp.Minimize(fit + self.mu * cp.normNuc(R))

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=self._solver, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CVXPY failed: {prob.status}")

        return torch.tensor(R.value, dtype=torch.complex64, device=x.device).unsqueeze(0)


def sample_covariance(x: torch.Tensor):
    """
    Calculates the sample covariance matrix.

    Args:
    -----
        X (torch.Tensor): Input samples matrix.

    Returns:
    --------
        Rx (torch.Tensor): Covariance matrix.
    """
    if x.dim() == 2:
        x = x[None, :, :]
    batch_size, sensor_number, samples_number = x.shape
    Rx = torch.einsum("bmt, btl -> bml", x, torch.conj(x).transpose(1, 2)) / samples_number
    return Rx


def get_cov_reconstruction_method(method: str, sys_model: SystemModel, **kwargs):
    method = method.lower()
    if not sys_model.is_sparse_array:
        method = "sample"

    if method == 'sample':
        if sys_model.params.signal_nature == 'coherent':
            recon = SpatialSmoothingReconstructor()
        else:
            recon = SampleCov()
    elif method == 'averaging':
        recon = AveragingReconstructor(sys_model)
    elif method == 'admm':
        recon = ADMMReconstructor(sys_model, **kwargs)
    elif method == 'admm_cvxpy':
        recon = ADMMReconstructorCVXPY(sys_model, **kwargs)
    else:
        raise ValueError(f"The covariance Reconstruction method: {method} isn't supported")

    return recon
