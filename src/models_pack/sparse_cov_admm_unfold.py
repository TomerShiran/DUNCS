from src.methods_pack.cov_reconstruct import sample_covariance
from src.models_pack.parent_model import ParentModel
from src.system_model import SystemModel
from src.utils import *
from src.methods_pack.music import MUSIC
from src.methods_pack.esprit import ESPRIT
from src.metrics.criterions import set_criterions, RMSPELoss


class DUNCS(ParentModel):
    def __init__(self, system_model: SystemModel, criterion, num_iterations: int, subspace_method: str, mu=1, rho=1):
        super().__init__(system_model, criterion)

        self.num_iter = num_iterations
        self.subspace_method = self.get_model_based_method(subspace_method, system_model)
        self.test_subspace_method =  self.subspace_method

        self.phi = build_phi(self.system_model.array)  # (|S|,|U|)
        self.phi_H = self.phi.t()

        self.criterion = set_criterions(criterion, self.system_model.array)[0]

        self.test_criterion = self.criterion
        self.test_iterations = self.num_iter

        self.U = self.phi.shape[1]  # |U|

        # mask diagonal  (P = Φᴴ Φ) → 1-D of length |U|²
        m = (self.phi_H @ self.phi).diag()  # (|U|,)
        self.P = (m[:, None] * m[None, :]).flatten()  # (|U|²,)

        # ---- Learned parameters ----
        self.rho_m = nn.Parameter(rho * torch.ones(self.num_iter, self.P.numel()))
        self.rho_r = nn.Parameter(rho * torch.ones(self.num_iter, self.P.numel()))
        self.tau = nn.Parameter((mu/rho) * torch.ones(self.num_iter, self.U))

        self.mu_u = nn.Parameter(torch.ones(self.num_iter, ))
        self.mu_v = nn.Parameter(torch.ones(self.num_iter, ))

    def get_learned_covariance(self, x: torch.Tensor, phase="train") -> torch.Tensor:
        Rx = sample_covariance(x)
        B, _, _ = Rx.shape
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
        # initial variables  (B,|U|,|U|)
        # ------------------------------------------------------------
        R = hermitian_proj(meas)
        S = R.clone()
        T = toeplitz_proj(R)
        Udual = torch.zeros_like(R)
        Vdual = torch.zeros_like(R)

        # ------------------------------------------------------------
        # ADMM iterations
        # ------------------------------------------------------------

        num_iter = self.test_iterations if phase == "test" else self.num_iter

        for k in range(num_iter):
            # R-update  (diagonal solve, batched)
            rhs = vec_meas + self.rho_r[k] * (S - Udual + T - Vdual).reshape(B, -1)

            inv_coeff = 1.0 / (self.P.to(dev, dtype) + 2 * self.rho_m[k])
            inv_coeff = inv_coeff.expand(B, -1)

            vec_R = inv_coeff * rhs
            R = vec_R.view(B, U, U)

            # S-update  (SVT)
            Z = R + Udual
            S = svt(Z, self.tau[k])

            # T-update  (Herm-Toeplitz-PSD)
            W = R + Vdual
            T = psd_proj(toeplitz_proj(hermitian_proj(W)))

            # dual ascent
            Udual += self.mu_u[k] * (R - S)
            Vdual += self.mu_v[k] * (R - T)

        return T

    def forward(self, x: torch.Tensor, num_sources: int, phase='train'):
        R = self.get_learned_covariance(x, phase)
        if phase == "train":
            doa_prediction, _, _ = self.subspace_method(R, num_sources)
        elif phase == "test":
            doa_prediction, _, _ = self.test_subspace_method(R, num_sources)
        else:
            raise NotImplementedError(f"Unknown phase {phase}")

        return doa_prediction

    def training_step(self, batch):
        x, sources_num, angles = self._prepare_batch(batch)
        if isinstance(self.criterion, RMSPELoss):
            doa_prediction = self(x, sources_num)
            loss = self.criterion(doa_prediction, angles)
        else:
            Rx = sample_covariance(x)
            R = self.get_learned_covariance(x)
            mu = torch.mean(self.tau[-1]) * torch.mean(self.rho_m[-1])
            loss = self.criterion(R, Rx, mu)

        return loss

    @torch.no_grad()
    def validation_step(self, batch):
        return self.training_step(batch)

    @torch.no_grad()
    def test_step(self, batch):
        x, sources_num, angles = self._prepare_batch(batch)
        if isinstance(self.test_criterion, RMSPELoss):
            doa_prediction = self(x, sources_num, phase='test')
            loss = self.test_criterion(doa_prediction, angles)
        else:
            Rx = sample_covariance(x)
            R = self.get_learned_covariance(x, phase='test')
            mu = torch.mean(self.tau[-1]) * torch.mean(self.rho_r[-1]).item()
            loss = self.test_criterion(R, Rx, mu)
        return loss

    @staticmethod
    def get_model_based_method(method_name: str, system_model: SystemModel):
        """
        Parameters
        ----------
        method_name(str): the method to use - music_1d, music_2d, root_music, esprit...
        system_model(SystemModel) : the system model to use as an argument to the method class.

        Returns
        -------
        an instance of the method.
        """
        if method_name.lower().endswith("music"):
            return MUSIC(system_model=system_model, estimation_parameter="angle")
        if method_name.lower().endswith("2d-music"):
            return MUSIC(system_model=system_model, estimation_parameter="angle, range")
        if method_name.lower().endswith("esprit"):
            return ESPRIT(system_model)

    def set_test_criteria(self, test_criterion):
        if isinstance(test_criterion, nn.Module):
            self.test_criterion = test_criterion
        else:
            self.test_criterion = \
                set_criterions(test_criterion, self.system_model.array)[0]

    def set_num_test_iterations(self, num_iter: int) -> None:
        if num_iter <= self.get_max_iterations():
            self.test_iterations = num_iter

    def get_max_iterations(self):
        return self.rho_m.shape[0]

    def set_test_subspace_method(self, subspace_method):
        self.test_subspace_method = self.get_model_based_method(subspace_method, self.system_model)