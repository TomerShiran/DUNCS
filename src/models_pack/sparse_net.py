from src.models_pack.subspacenet import SubspaceNet
from src.utils import *


class SparseNet(SubspaceNet):

    def __init__(self, **kwargs):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.


        """
        super().__init__(**kwargs)
        if self.system_model.is_sparse_array:
            self.L = len(self.system_model.virtual_array_ula_seg)
        else:
            self.L = len(self.system_model.array)

        self.differences_array = self.system_model.array[:, None] - self.system_model.array[None, :]

    def pre_processing(self, x):
        """
        The input data is a complex signal of size [batch, N, T] and the input to the model supposed to be complex
        tensors of size [batch, tau, 2N_virtual, N_virtual]. The input is the auto-correlation matrix, created as the
        covariance matrix is generated in the paper: "Remarks on the Spatial Smoothing Step in Coarray MUSIC"
        """

        batch_size = x.shape[0]
        Rx_tau = torch.zeros(batch_size, self.tau, 2 * self.L, self.L, device=device)
        meu = torch.mean(x, dim=-1, keepdim=True).to(device)
        center_x = x - meu
        if center_x.dim() == 2:
            center_x = center_x[None, :, :]

        for i in range(self.tau):
            x1 = center_x[:, :, :center_x.shape[-1] - i].to(torch.complex128)
            x2 = torch.conj(center_x[:, :, i:]).transpose(1, 2).to(torch.complex128)
            Rx_lag = torch.einsum("BNT, BTM -> BNM", x1, x2) / (center_x.shape[-1] - i - 1)
            x_s_diff = torch.zeros(batch_size, 2 * self.L - 1, dtype=torch.complex128, device=device)
            for j, lag in enumerate(range(-np.max(self.system_model.virtual_array_ula_seg), np.max(self.system_model.virtual_array_ula_seg) + 1)):
                pairs = torch.from_numpy(self.differences_array) == lag
                if pairs.any():
                    x_s_diff[:, j] = torch.mean(Rx_lag[:, pairs], dim=1)

            for j in range(self.L):
                start_idx = self.L - 1 - j
                Rx_col = torch.cat((torch.real(x_s_diff[:, start_idx:start_idx + self.L]),
                                    torch.imag(x_s_diff[:, start_idx:start_idx + self.L])), dim=1)
                Rx_tau[:, i, :, j] = Rx_col

        return Rx_tau

    def get_model_file_name(self):
        M = str(self.system_model.params.M).replace(' ', '-')
        snr = str(self.system_model.params.snr).replace(' ', '-')
        return f"{self.get_model_name()}_" + \
               f"N={self.N}_" + \
               f"L={self.L}_" + \
               f"M={M}_" + \
               f"T={self.system_model.params.T}_" + \
               f"{self.system_model.params.signal_type}_" + \
               f"SNR={snr}_" + \
               f"{self.system_model.params.field_type}_field_" + \
               f"{self.system_model.params.signal_nature}_" + \
               f"eta={self.system_model.params.eta}_" + \
               f"sv_var={self.system_model.params.sv_noise_var}"
