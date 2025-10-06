"""
SubspaceNet: model-based deep learning algorithm as described in:
        [2] "SubspaceNet: Deep Learning-Aided Subspace methods for DoA Estimation".
"""

from src.models_pack.parent_model import ParentModel
from src.system_model import SystemModel
from src.utils import *
from src.metrics.criterions import EigenRegularizationLoss

from src.methods_pack.music import MUSIC
from src.methods_pack.esprit import ESPRIT
from src.methods_pack.root_music import RootMusic
from src.metrics.criterions import set_criterions

class SubspaceNet(ParentModel):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        N (int):
        diff_method (str):
        field_type (str):
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        ReLU (nn.ReLU): ReLU activation function.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, diff_method: str = "root_music",
                 system_model: SystemModel = None, field_type: str = "Far", eigen_regularization_weight=0,
                 criterion="rmspe"):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__(system_model, criterion)
        self.criterion = set_criterions(criterion.lower())[0]
        self.tau = tau
        self.N = self.system_model.params.N
        self.diff_method = None
        self.field_type = field_type
        self.p = 0.15
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(self.p)
        self.ReLU = nn.ReLU()
        self.norm1 = SpectralNormalization()
        self._eigen_regularization = EigenRegularizationLoss(eigen_regularization_weight)
        # Set the subspace method for training
        self.set_diff_method(diff_method, system_model)

    def get_learned_covariance(self, X: torch.Tensor) -> torch.Tensor:
        """
            This function is the "real" forward pass of the SubspaceNet.
            It receives the input tensor and returns the surrogate covariance matrix.
            Args:
                x: the input tensor of shape [Batch size, N, T]

            Returns:
                Rz: the surrogate covariance matrix of shape [Batch size, N, N]
        """
        x = self.pre_processing(X)
        # Rx_tau shape: [Batch size, tau, 2N, N]
        N = x.shape[-1]
        self.batch_size = x.shape[0]
        ############################
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(x)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)

        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #4
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, :N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        Kx_tag = self.norm1(Kx_tag)
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        return Rz

    def forward(self, x: torch.Tensor, sources_num: torch.tensor = None, known_angles: torch.tensor = None):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            x (torch.Tensor): Input tensor of shape [Batch size, N, T].
            sources_num (torch.Tensor): The number of sources in the signal.
            known_angles (torch.Tensor): The known angles for the near-field scenario.

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """

        # Feed surrogate covariance to the differentiable subspace algorithm
        Rz = self.get_learned_covariance(x)

        if self.field_type == "Far":
            method_output = self.diff_method(Rz, sources_num)
            if isinstance(self.diff_method, RootMusic):
                doa_prediction, doa_all_predictions, roots = method_output
                return doa_prediction, doa_all_predictions, roots
            elif isinstance(self.diff_method, ESPRIT):
                # Esprit output
                doa_prediction, sources_estimation, eigen_regularization = method_output
                return doa_prediction, sources_estimation, eigen_regularization
            elif isinstance(self.diff_method, MUSIC):
                doa_prediction, sources_estimation, eigen_regularization = method_output
                return doa_prediction, sources_estimation, eigen_regularization
            else:
                raise Exception(f"SubspaceNet.forward: Method {self.diff_method} is not defined for SubspaceNet")

        elif self.field_type == "Near":
            if known_angles is None:
                predictions, sources_estimation, eigen_regularization = self.diff_method(
                    Rz, number_of_sources=sources_num)
                doa_prediction, distance_prediction = predictions
                return doa_prediction, distance_prediction, sources_estimation, eigen_regularization
            else:  # the angles are known
                distance_prediction = self.diff_method(
                    cov=Rz, number_of_sources=sources_num, known_angles=known_angles)
                if isinstance(distance_prediction, tuple):
                    distance_prediction, _, _ = distance_prediction
                return known_angles, distance_prediction, Rz

    def pre_processing(self, x):
        """
        The input data is a complex signal of size [batch, N, T] and the input to the model supposed to be complex
         tensors of size [batch, tau, 2N, N].
        """
        batch_size = x.shape[0]
        Rx_tau = torch.zeros(batch_size, self.tau, 2 * self.N, self.N, device=device)
        meu = torch.mean(x, dim=-1, keepdim=True).to(device)
        center_x = x - meu
        if center_x.dim() == 2:
            center_x = center_x[None, :, :]

        for i in range(self.tau):
            x1 = center_x[:, :, :center_x.shape[-1] - i].to(torch.complex128)
            x2 = torch.conj(center_x[:, :, i:]).transpose(1, 2).to(torch.complex128)
            Rx_lag = torch.einsum("BNT, BTM -> BNM", x1, x2) / (center_x.shape[-1] - i - 1)
            Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), dim=1)
            Rx_tau[:, i, :, :] = Rx_lag

        return Rx_tau

    def set_diff_method(self, diff_method: str, system_model):
        """Sets the differentiable subspace method for training subspaceNet.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNet
        """
        if self.field_type == "Far":
            if diff_method.startswith("root_music"):
                self.diff_method = RootMusic(system_model=system_model)
            elif diff_method.startswith("esprit"):
                self.diff_method = ESPRIT(system_model=system_model)
            elif diff_method.startswith("music_1D"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="angle")
            else:
                raise Exception(f"SubspaceNet.set_diff_method:"
                                f" Method {diff_method} is not defined for SubspaceNet in "
                                f"{self.field_type} scenario")
        elif self.field_type == "Near":
            if diff_method.startswith("music_2D"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="angle, range")
            elif diff_method.startswith("music_1D"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="range")
            else:
                raise Exception(f"SubspaceNet.set_diff_method:"
                                f" Method {diff_method} is not defined for SubspaceNet in "
                                f"{self.field_type} Field scenario")

    def anti_rectifier(self, X):
        """Applies the anti-rectifier operation to the input tensor.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor after applying the anti-rectifier operation.

        """
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def adjust_diff_method_temperature(self, epoch):
        if isinstance(self.diff_method, MUSIC):
            if epoch % 20 == 0 and epoch != 0:
                self.diff_method.adjust_cell_size()
                print(f"Model temepartue updated --> {self.get_diff_method_temperature()}")

    def get_diff_method_temperature(self):
        if isinstance(self.diff_method, MUSIC):
            if self.diff_method.estimation_params in ["angle", "range"]:
                return self.diff_method.cell_size
            elif self.diff_method.estimation_params == "angle, range":
                return {"angle_cell_size": self.diff_method.cell_size_angle,
                        "distance_cell_size": self.diff_method.cell_size_distance}

    def get_model_params(self):
        tau = self.tau
        diff_method = self.diff_method
        return f"tau={tau}_diff_method={diff_method}"

    def training_step(self, batch):
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, sources_estimation, eigen_regularization = self(x, sources_num)
        loss = self.criterion(doa_prediction, angles)
        acc = self._eigen_regularization.source_estimation_accuracy(sources_num, sources_estimation)
        loss = self._eigen_regularization.get_regularized_loss(loss, eigen_regularization)
        return loss, acc, eigen_regularization

    def validation_step(self, batch):
        x, sources_num, angles = self._prepare_batch(batch)
        doa_prediction, source_estimation, eigen_regularization = self(x, sources_num)
        loss = self.criterion(doa_prediction, angles)
        acc = self._eigen_regularization.source_estimation_accuracy(sources_num, source_estimation)
        return loss, acc

    def test_step(self, batch):
        return self.validation_step(batch)
