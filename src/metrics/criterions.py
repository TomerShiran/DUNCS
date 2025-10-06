"""Subspace-Net 
Details
----------
Name: criterions.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 03/06/23

Purpose:
--------
The purpose of this script is to define and document several loss functions (RMSPELoss and MSPELoss)
and a helper function (permute_prediction) for calculating the Root Mean Square Periodic Error (RMSPE)
and Mean Square Periodic Error (MSPE) between predicted values and target values.
The script also includes a utility function RMSPE and MSPE that calculates the RMSPE and MSPE values
for numpy arrays.

This script includes the following Classes anf functions:

* permute_prediction: A function that generates all possible permutations of a given prediction tensor.
* RMSPELoss (class): A custom PyTorch loss function that calculates the RMSPE loss between predicted values
    and target values. It inherits from the nn.Module class and overrides the forward method to perform
    the loss computation.
* MSPELoss (class): A custom PyTorch loss function that calculates the MSPE loss between predicted values
  and target values. It inherits from the nn.Module class and overrides the forward method to perform the loss computation.
* RMSPE (function): A function that calculates the RMSPE value between the DOA predictions and target DOA values for numpy arrays.
* MSPE (function): A function that calculates the MSPE value between the DOA predictions and target DOA values for numpy arrays.
* set_criterions(function): Set the loss criteria based on the criterion name.

"""

import numpy as np
import torch.nn as nn
import torch
from itertools import permutations
from typing import List

from src.utils import *
from scipy.optimize import linear_sum_assignment
import time
BALANCE_FACTOR = 1.0


def add_line_to_file(file_name, line_to_add):
    try:
        with open(file_name, 'r+') as file:
            lines = file.readlines()
            if not lines or lines[-1].strip() != line_to_add:
                file.write('\n' + line_to_add)
                # print(f"Added line '{line_to_add}' to the file.")
            else:
                pass
                # print(f"Line '{line_to_add}' already exists in the file.")
    except FileNotFoundError:
        with open(file_name, 'w') as file:
            file.write(line_to_add)
            # print(f"Created file '{file_name}' with line '{line_to_add}'.")


def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])
        
    """
    torch_perm_list = []
    prediction = torch.atleast_1d(prediction)
    for p in list(permutations(range(prediction.shape[0]), prediction.shape[0])):
        torch_perm_list.append(prediction.index_select(0, torch.tensor(list(p), dtype=torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim=0)
    return predictions


class RMSELoss(nn.MSELoss):
    def __init__(self, *args):
        super(RMSELoss, self).__init__(*args)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = super(RMSELoss, self).forward(input.to(device), target.to(device))
        return torch.sqrt(mse_loss)


class RMSPELoss(nn.Module):
    """Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """

    def __init__(self):
        super(RMSPELoss, self).__init__()

    def forward(self, doa_predictions: torch.Tensor, doa_targets: torch.Tensor):
        """
        Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa,
        and possibly distance_predictions and distance.
        The predicted values and target values are expected to be in radians for the DOA values.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSE.
        The minimum RMSE value among all permutations is selected for each sample,
         including the RMSE for the distance values with the same permutation.
        Finally, the method averged the RMSE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).
            distance_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            The default value is None.
            distance (torch.Tensor): Target values tensor of shape (batch_size, num_targets).The default value is None.


        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        B, num_sources = doa_predictions.shape

        # Compute the pairwise cost matrix for angles (B, num_sources, num_sources)
        diff_matrix_angle = self.compute_modulo_error(doa_predictions.unsqueeze(2), doa_targets.unsqueeze(1))
        cost_matrix_angle = diff_matrix_angle ** 2

        # Use the loop-based assignment
        assignments = self.batch_hungarian_assignments(cost_matrix_angle)  # shape: (B, num_sources)

        batch_indices = torch.arange(B, device=doa_predictions.device).unsqueeze(1).expand(B, num_sources)
        row_indices = torch.arange(num_sources, device=doa_predictions.device).unsqueeze(0).expand(B, num_sources)
        optimal_angle_errors = diff_matrix_angle[batch_indices, row_indices, assignments]
        rmspe_angle = torch.sqrt(torch.sum(optimal_angle_errors ** 2, dim=1) / num_sources)


        total_loss = torch.sum(rmspe_angle)
        return total_loss

    @staticmethod
    def batch_hungarian_assignments(cost_matrices):
        """
        cost_matrices: Tensor of shape (B, n, n)
        Returns:
            Tensor of shape (B, n) containing the assignment (i.e. permutation indices)
        """
        assignments = []
        B = cost_matrices.shape[0]
        for i in range(B):
            # Force a real tensor with its own storage:
            cost_matrix = cost_matrices[i].clone().detach().cpu()
            cost_np = cost_matrix.contiguous().numpy()
            _, col_ind = linear_sum_assignment(cost_np)
            assignments.append(torch.tensor(col_ind, device=cost_matrices.device))
        return torch.stack(assignments, dim=0)

    @staticmethod
    def compute_modulo_error(pred, target):
        """
        Compute the error with modulo pi so that the differences are in [-pi/2, pi/2].

        Args:
            pred (torch.Tensor): Tensor of shape (..., 1) or (..., n, 1)
            target (torch.Tensor): Tensor of shape (..., 1) or (..., 1, n)

        Returns:
            torch.Tensor: The error tensor broadcasted to shape (..., n, n) if pred and target are unsqueezed.
        """
        diff = pred - target  # Broadcasting happens here.
        diff = (diff + torch.pi / 2) % torch.pi - torch.pi / 2
        return diff


class MSPELoss(nn.Module):
    """Mean Square Periodic Error (MSPE) loss function.
    This loss function calculates the MSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the MSPE loss between the predictions and target values.

    Example:
        criterion = MSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """

    def __init__(self):
        super(MSPELoss, self).__init__()

    def forward(self, doa_predictions: torch.Tensor, doa,
                distance_predictions: torch.Tensor = None, distance: torch.Tensor = None):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).
            distance_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            The default value is None.
            distance (torch.Tensor): Target values tensor of shape (batch_size, num_targets).The default value is None.

        Returns:
            torch.Tensor: The computed MSPE loss.

        Raises:
            None
        """
        rmspe = []
        for iter in range(doa_predictions.shape[0]):
            rmspe_list = []
            batch_predictions = doa_predictions[iter].to(device)
            targets = doa[iter].to(device)
            prediction_perm = permute_prediction(batch_predictions).to(device)
            for prediction in prediction_perm:
                # Calculate error with modulo pi
                error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2
                # Calculate MSE over all permutations
                rmspe_val = (1 / len(targets)) * (torch.linalg.norm(error) ** 2)
                rmspe_list.append(rmspe_val)
            rmspe_tensor = torch.stack(rmspe_list, dim=0)
            rmspe_min = torch.min(rmspe_tensor)
            # Choose minimal error from all permutations
            rmspe.append(rmspe_min)
        result = torch.sum(torch.stack(rmspe, dim=0))

        if distance_predictions is not None:
            if distance is None:
                raise Exception("Target distances values are missing!")
            mse_loss = nn.MSELoss()
            distance_loss = mse_loss(distance_predictions, distance)
            result += distance_loss
        return result


class CartesianLoss(nn.Module):
    def __init__(self):
        super(CartesianLoss, self).__init__()

    def forward(self, predictions_angle: torch.Tensor, targets_angle: torch.Tensor, predictions_distance: torch.Tensor,
                targets_distance: torch.Tensor):
        """
        the input given is expected to contain angels and distances.
        """
        M = targets_angle.shape[1]
        if predictions_angle.shape[1] > targets_angle.shape[1]:
            # in this case, randomly drop some of the predictions
            indices = torch.randperm(predictions_angle.shape[1])[:M].to(device)
            predictions_angle = torch.gather(predictions_angle, 1, indices[None, :])
            predictions_distance = torch.gather(predictions_distance, 1, indices[None, :])

        elif predictions_angle.shape[1] < targets_angle.shape[1]:
            # add a random angle to the predictions
            random_angles = torch.distributions.uniform.Uniform(-torch.pi / 3, torch.pi / 3).sample([predictions_angle.shape[0], M - predictions_angle.shape[1]])
            random_ranges = torch.distributions.uniform.Uniform(torch.min(targets_distance).item(), torch.max(targets_distance).item()).sample([predictions_angle.shape[0], M - predictions_angle.shape[1]])
            predictions_angle = torch.cat((predictions_angle, random_angles.to(device)), dim=1)
            predictions_distance = torch.cat((predictions_distance, random_ranges.to(device)), dim=1)

        number_of_samples = predictions_angle.shape[0]
        true_x = torch.cos(targets_angle) * targets_distance
        true_y = torch.sin(targets_angle) * targets_distance
        coords_true = torch.stack((true_x, true_y), dim=2)
        pred_x = torch.cos(predictions_angle) * predictions_distance
        pred_y = torch.sin(predictions_angle) * predictions_distance
        coords_pred = torch.stack((pred_x, pred_y), dim=2)
        # need to consider all possible permutations for M sources
        perm = list(permutations(range(M), M))
        perm = torch.tensor(perm, dtype=torch.int64).to(device)
        num_of_perm = len(perm)

        error = torch.tile(coords_true[:, None, :, :], (1, num_of_perm, 1, 1)) - coords_pred[:, perm]
        loss = torch.sqrt(torch.sum(error ** 2, dim=-1))
        loss = torch.mean(loss, dim=-1)
        loss = torch.min(loss, dim=-1)
        return torch.sum(loss[0])
        # loss = []
        # for batch in range(number_of_samples):
        #     loss_per_sample = []
        #     for p in perm:
        #         loss_per_sample.append(torch.sqrt(torch.sum((coords_true[batch] - coords_pred[batch, p, :]) ** 2, dim=1)).mean())
        #     loss.append(torch.min(torch.stack(loss_per_sample, dim=0)))
        # if (loss_[0] != torch.stack(loss, dim=0)).all():
        #     raise ValueError("Error in Cartesian Loss")


class ADMMObjective(nn.Module):
    r"""
    Unsupervised loss for training an unfolded-ADMM reconstructor:

        ℒ(R̂; Rx) = ‖Φ R̂ Φᴴ − Rx‖_F²  +  μ · ‖R̂‖_* .

    * **data-fit term** forces R̂ to agree with the measured sub-array
      covariance.
    * **nuclear-norm term** keeps R̂ low-rank (signal subspace).

    Parameters
    ----------
    system_model : SystemModel
        Provides `.array` (physical indices) and `.virtual_array`
        (contiguous co-array indices) so we can build Φ.
    mu : float
        Weight on the nuclear-norm term.
    """

    def __init__(self, array, mu: float = 2.5e-3) -> None:
        super().__init__()
        self.mu = mu
        self.phi = build_phi(array)
        self.phi_H = self.phi.t()


    def forward(self,
                R_hat: Tensor,    # (B, |U|, |U|)
                Rx:     Tensor,    # (B, |S|, |S|)
                mu=None) -> Tensor:
        """
        Returns a *scalar* loss (mean over batch).

        Both tensors must share the same `dtype` / `device`.
        """
        phi  = self.phi.to(R_hat)
        phi_H = self.phi_H.to(R_hat)

        # data-fit Frobenius term
        residual = phi @ R_hat @ phi_H - Rx       # (B, |S|, |S|)
        data_fit = residual.flatten(1).norm(dim=1, p=2).pow(2)  # (B,)

        # nuclear norm  ‖R̂‖_*
        _, s, _ = torch.linalg.svd(R_hat, full_matrices=False)
        nuc = s.sum(dim=1)                       # (B,)

        mu = self.mu if mu is None else mu
        loss = data_fit + mu * nuc
        return loss.sum()                       # scalar


def set_criterions(criterions_name: List[str], *args):
    """
    Set the loss criteria based on the criterion name.

    Parameters:
        criterions_name (List[str]): Names of the criterions.

    Returns:
        criterion (nn.Module): Loss criterion for model evaluation.
        subspace_criterion (Callable): Loss criterion for subspace method evaluation.

    Raises:
        Exception: If the criterion name is not defined.
    """
    criterions = []
    criterions_name = [criterions_name] if isinstance(criterions_name, str) else criterions_name

    for name in criterions_name:
        if name.startswith("rmspe"):
            criterion = RMSPELoss()
        elif name.startswith("mse"):
            criterion = nn.MSELoss()
        elif name.startswith("rmse"):
            criterion = RMSPELoss()
        elif name.startswith("cartesian"):
            criterion = CartesianLoss()
        elif name == "admm_objective":
            criterion = ADMMObjective(*args)
        else:
            raise Exception(f"criterions.set_criterions: Criterion {name} is not defined")

        criterions.append(criterion)

    if len(criterions) == 0:
        raise ValueError("At least one criterion must be used")

    return criterions


class EigenRegularizationLoss:
    EIGEN_REGULARIZATION_WEIGHT = 0

    def __init__(self, init_value=EIGEN_REGULARIZATION_WEIGHT):
        self._eigenregularization_weight = init_value

    def get_eigenregularization_weight(self):
        return self._eigenregularization_weight

    def source_estimation_accuracy(self, sources_num, source_estimation=None):
        if source_estimation is None:
            return 0
        return torch.sum(source_estimation == sources_num * torch.ones_like(source_estimation).float()).item()

    def get_regularized_loss(self, loss, l_eig=None):
        if l_eig is not None:
            loss_r = loss + self._eigenregularization_weight * l_eig
        else:
            loss_r = loss
        return torch.sum(loss_r)



if __name__ == "__main__":
    prediction = torch.tensor([1, 2, 3])
    print(permute_prediction(prediction))
