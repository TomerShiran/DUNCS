"""
Subspace-Net

Details
----------
Name: training.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This code provides functions for training and simulating the Subspace-Net model.

Classes:
----------
- TrainingParams: A class that encapsulates the training parameters for the model.

Methods:
----------
- train: Function for training the model.
- train_model: Function for performing the training process.
- plot_learning_curve: Function for plotting the learning curve.
- simulation_summary: Function for printing a summary of the simulation parameters.

Attributes:
----------
None
"""
# Imports
import matplotlib.pyplot as plt
import copy
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import random_split, Dataset
import math

# internal imports
from src.metrics.criterions import *
from src.system_model import SystemModelParams
from src.models import (ModelGenerator)
from src.evaluation import evaluate_dnn_model
from src.data_handler import collate_fn, SameLengthBatchSampler


class TrainingParams(object):
    """
    A class that encapsulates the training parameters for the model.

    Methods
    -------
    - __init__: Initializes the TrainingParams object.
    - set_batch_size: Sets the batch size for training.
    - set_epochs: Sets the number of epochs for training.
    - set_model: Sets the model for training.
    - load_model: Loads a pre-trained model.
    - set_optimizer: Sets the optimizer for training.
    - set_schedular: Sets the scheduler for learning rate decay.
    - set_criterion: Sets the loss criterion for training.
    - set_training_dataset: Sets the training dataset for training.

    Raises
    ------
    Exception: If the model type is not defined.
    Exception: If the optimizer type is not defined.
    """

    def __init__(self):
        """
        Initializes the TrainingParams object.
        """
        self.criterion = None
        self.model = None
        self.diff_method = None
        self.tau = None
        self.model_type = None
        self.epochs = None
        self.batch_size = None
        self.training_objective = None

    def set_training_objective(self, training_objective: str):
        """

        Args:
            training_objective:

        Returns:

        """
        if training_objective.lower() == "angle":
            self.training_objective = "angle"
        elif training_objective.lower() == "range":
            self.training_objective = "range"
        elif training_objective.lower() == "angle, range":
            self.training_objective = "angle, range"
        elif training_objective.lower() == "source_estimation":
            self.training_objective = "source_estimation"
        else:
            raise Exception(f"TrainingParams.set_training_objective:"
                            f" Unrecognized training objective : {training_objective}.")
        return self

    def set_batch_size(self, batch_size: int):
        """
        Sets the batch size for training.

        Args
        ----
        - batch_size (int): The batch size.

        Returns
        -------
        self
        """
        self.batch_size = batch_size
        return self

    def set_epochs(self, epochs: int):
        """
        Sets the number of epochs for training.

        Args
        ----
        - epochs (int): The number of epochs.

        Returns
        -------
        self
        """
        self.epochs = epochs
        return self

    def set_model(self, model_gen: ModelGenerator = None):
        """
        Sets the model for training.

        Args
        ----
        - model_gen (ModelGenerator): The system model object.

        Returns
        -------
        self

        Raises
        ------
        Exception: If the model type is not defined.
        """
        # assign model to device
        self.model = model_gen.model.to(device)
        return self

    def set_optimizer(self, optimizer: str, learning_rate: float, weight_decay: float):
        """
        Sets the optimizer for training.

        Args
        ----
        - optimizer (str): The optimizer type.
        - learning_rate (float): The learning rate.
        - weight_decay (float): The weight decay value (L2 regularization).

        Returns
        -------
        self

        Raises
        ------
        Exception: If the optimizer type is not defined.
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # Assign optimizer for training
        if optimizer.startswith("Adam"):
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer.startswith("AdamW"):
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer.startswith("SGD"):
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer == "SGD Momentum":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=0.9
            )
        else:
            raise Exception(
                f"TrainingParams.set_optimizer: Optimizer {optimizer} is not defined"
            )
        return self

    def set_schedular(self, scheduler, step_size: int, gamma: float, total_steps: int):
        """
        Sets the scheduler for learning rate decay.

        Args:
        ----------
        - step_size (float): Number of steps for learning rate decay iteration.
        - gamma (float): Learning rate decay value.

        Returns:
        ----------
        self
        """
        if scheduler == "StepLR":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler == "ReduceLROnPlateau":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=gamma,
                                                  patience=7, threshold=7e-4)
        elif scheduler == "OneCycleLR":
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=2e-4, pct_start=0.55,
                                                           total_steps=int(total_steps), div_factor=100,
                                                           final_div_factor=2000, cycle_momentum=False)
        elif scheduler == "CustomLR":
            warmup_steps = int(0.15 * total_steps)

            sched_warmup = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
            sched_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps, eta_min=self.learning_rate * 0.02)

            self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[sched_warmup, sched_cosine],
                                     milestones=[warmup_steps])

        else:
            raise ValueError(f"Scheduler {scheduler} is not defined.")

        # Number of steps for learning rate decay iteration
        self.step_size = step_size
        # learning rate decay value
        self.gamma = gamma
        return self


    def set_training_dataset(self, train_dataset):
        """
        Sets the training dataset for training.

        Args
        ----
        - train_dataset (list): The training dataset.

        Returns
        -------
        self
        """
        # Divide into training and validation datasets
        train_size = int(0.9 * len(train_dataset))
        valid_size = len(train_dataset) - train_size

        train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

        # init sampler
        batch_sampler_train = SameLengthBatchSampler(train_dataset, batch_size=self.batch_size)
        batch_sampler_valid = SameLengthBatchSampler(valid_dataset, batch_size=128, shuffle=False)
        # Transform datasets into DataLoader objects
        self.train_dataset = torch.utils.data.DataLoader(
            train_dataset,collate_fn=collate_fn, batch_sampler=batch_sampler_train, pin_memory=True,
        )
        self.valid_dataset = torch.utils.data.DataLoader(
            valid_dataset,collate_fn=collate_fn, batch_sampler=batch_sampler_valid, pin_memory=True,
        )
        return self


class EarlyStopping:
    def __init__(self, mode="min", patience=10, min_delta=1e-4, restore_best=True):
        assert mode in {"min","max"}
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best = math.inf if mode == "min" else -math.inf
        self.bad_epochs = 0
        self.best_state = None

    def _improved(self, value):
        if self.mode == "min":
            return (self.best - value) > self.min_delta
        else:
            return (value - self.best) > self.min_delta

    def step(self, value, model=None):
        """Return True if training should stop."""
        if self._improved(value):
            self.best = value
            self.bad_epochs = 0
            if self.restore_best and model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)



def train(
        training_parameters: TrainingParams,
        plot_curves: bool = True,
        saving_path: Path = None,
        save_figures: bool = False,
):
    """
    Wrapper function for training the model.

    Args:
    ----------
    - training_params (TrainingParams): An instance of TrainingParams containing the training parameters.
    - model_name (str): The name of the model.
    - plot_curves (bool): Flag to indicate whether to plot learning and validation loss curves. Defaults to True.
    - saving_path (Path): The directory to save the trained model.

    Returns:
    ----------
    model: The trained model.
    loss_train_list: List of training loss values.
    loss_valid_list: List of validation loss values.

    Raises:
    ----------
    Exception: If the model type is not defined.
    Exception: If the optimizer type is not defined.
    """
    # Set the seed for all available random operations
    # set_unified_seed()
    # Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)
    # Train the model
    train_res = train_model(training_parameters,
                            checkpoint_path=saving_path)
    model = train_res.get("model")
    loss_train_list = train_res.get("loss_train_list")
    loss_valid_list = train_res.get("loss_valid_list")
    loss_train_list_angles = train_res.get("loss_train_list_angles")
    loss_train_list_ranges = train_res.get("loss_train_list_ranges")
    loss_valid_list_angles = train_res.get("loss_valid_list_angles")
    loss_valid_list_ranges = train_res.get("loss_valid_list_ranges")
    acc_train_list = train_res.get("acc_train_list")
    acc_valid_list = train_res.get("acc_valid_list")
    reg_loss_train_list = train_res.get("reg_loss_train_list")

    figures_saving_path = Path(saving_path).parent / "simulations" / "results" / "plots"
    if plot_curves:
        if acc_train_list is not None and acc_valid_list is not None:
            fig_acc = plot_accuracy_curve(
                list(range(1, len(acc_train_list) + 1)), acc_train_list, acc_valid_list,
                model_name=model._get_name()
            )
            if save_figures:
                fig_acc.savefig(figures_saving_path / f"Accuracy_{model.get_model_name()}_{dt_string_for_save}.png")
            fig_acc.show()
        fig_loss = plot_learning_curve(
            list(range(1, len(loss_valid_list) + 1)), loss_train_list, loss_valid_list,
            model_name=model._get_name(),
            angle_train_loss=loss_train_list_angles,
            angle_valid_loss=loss_valid_list_angles,
            range_train_loss=loss_train_list_ranges,
            range_valid_loss=loss_valid_list_ranges
        )

        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, len(reg_loss_train_list) + 1), reg_loss_train_list, label="eigen training Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("Eigen Regularization Loss")
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        if save_figures:
            fig_loss.savefig(figures_saving_path / f"Loss_{model.get_model_name()}_{dt_string_for_save}.png")
        fig_loss.show()

    # Save models best weights
    torch.save(model.state_dict(), saving_path / model.get_model_file_name())
    # Plot learning and validation loss curves
    return model, loss_train_list, loss_valid_list


def train_model(training_params: TrainingParams, checkpoint_path=None) -> dict:
    """
    Function for training the model.

    Args:
    -----
        training_params (TrainingParams): An instance of TrainingParams containing the training parameters.
        model_name (str): The name of the model.
        checkpoint_path (str): The path to save the checkpoint.

    Returns:
    --------
        model: The trained model.
        loss_train_list (list): List of training losses per epoch.
        loss_valid_list (list): List of validation losses per epoch.
    """
    # Initialize model and optimizer
    model = training_params.model
    model = model.to(device)
    optimizer = training_params.optimizer
    # Initialize losses
    loss_train_list = []
    loss_valid_list = []
    acc_train_list = []
    acc_valid_list = []
    reg_loss_train_list = []
    min_valid_loss = np.inf

    # Set initial time for start training
    since = time.time()
    print("\n---Start Training Stage ---\n")
    # Run over all epochs

    total_batches = len(training_params.train_dataset)
    total_iterations = training_params.epochs * total_batches  # Total number of batches across all epochs
    # torch.autograd.set_detect_anomaly(True)

    # early = EarlyStopping(mode="min", patience=15, min_delta=1e-4, restore_best=True)
    # Initialize tqdm once for the entire training process
    with tqdm(total=total_iterations, desc="Total Training Progress", unit="batch") as pbar:
        for epoch in range(training_params.epochs):
            epoch_train_loss = 0.0
            epoch_train_reg_loss = 0.0
            epoch_train_acc = 0.0

            model.train()
            train_length = 0

            for data in training_params.train_dataset:
                # reset gradients
                optimizer.zero_grad()

                # Forward pass
                loss = model.training_step(data)
                if isinstance(loss, tuple):
                    loss, acc, eigen_regularization = loss
                else:
                    acc, eigen_regularization = None, None

                epoch_train_loss += loss.item()
                if acc is not None:
                    epoch_train_acc += acc
                if eigen_regularization is not None:
                    epoch_train_reg_loss += torch.sum(eigen_regularization).item()

                train_length += data[0].shape[0]

                try:
                    loss.backward()  # retain_graph=True
                except RuntimeError as r:
                    print(f"linalg error: \n{r}")

                else:
                    # optimizer update
                    optimizer.step()
                    if isinstance(training_params.scheduler, (lr_scheduler.OneCycleLR, lr_scheduler.SequentialLR)):
                        training_params.scheduler.step()

                pbar.update(1)

            ####################################################################################
            epoch_train_loss /= train_length
            epoch_train_reg_loss /= train_length
            epoch_train_acc /= train_length
            # End of epoch. Calculate the average loss
            loss_train_list.append(epoch_train_loss)
            reg_loss_train_list.append(epoch_train_reg_loss)

            # Calculate evaluation loss
            valid_loss = evaluate_dnn_model(model, training_params.valid_dataset, mode="valid")
            loss_valid_list.append(valid_loss.get("loss"))

            # Update scheduler
            if isinstance(training_params.scheduler, lr_scheduler.ReduceLROnPlateau):
                training_params.scheduler.step(loss_valid_list[-1])

            elif isinstance(training_params.scheduler, lr_scheduler.StepLR):
                training_params.scheduler.step()

            elif not isinstance(training_params.scheduler, (lr_scheduler.OneCycleLR, lr_scheduler.SequentialLR)):
                raise NotImplementedError("update Step isn't implemented for this scheduler")

            # Report results
            result_txt = (f"[Epoch : {epoch + 1}/{training_params.epochs}]"
                          f" Train loss = {epoch_train_loss:.6f}, Validation loss = {valid_loss.get('loss'):.6f}")

            acc_train_list.append(epoch_train_acc * 100)
            acc_valid_list.append(valid_loss.get('Accuracy') * 100)
            result_txt += (f"\nAccuracy for sources estimation: Train = {100 * epoch_train_acc:.2f}%, "
                           f"Validation = {valid_loss.get('Accuracy') * 100:.2f}%")
            result_txt += f"\nlr {training_params.scheduler.get_last_lr()[0]}"

            print(result_txt)
            # Save best model weights
            if min_valid_loss > valid_loss.get("loss"):
                print(
                    f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss.get('loss'):.6f}) \t Saving The Model"
                )
                min_valid_loss = valid_loss.get("loss")
                best_epoch = epoch
                # Saving State Dict
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path / model.get_model_file_name())

            # if early.step(valid_loss.get("loss"), model):
            #     print("Early Stopping, plateau reached, epoch:", epoch)
            #     break

    # Training complete
    time_elapsed = time.time() - since
    print("\n--- Training summary ---")
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Minimal Validation loss: {min_valid_loss:4f} at epoch {best_epoch}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path / model.get_model_file_name())
    res = {"model": model, "loss_train_list": loss_train_list, "loss_valid_list": loss_valid_list,
           "reg_loss_train_list": reg_loss_train_list}
    if len(acc_train_list) > 0 and len(acc_valid_list) > 0:
        res["acc_train_list"] = acc_train_list
        res["acc_valid_list"] = acc_valid_list
    return res


def plot_accuracy_curve(epoch_list, train_acc: list, validation_acc: list, model_name: str = None):
    """
    Plot the learning curve.

    Args:
    -----
        epoch_list (list): List of epochs.
        train_loss (list): List of training losses per epoch.
        validation_loss (list): List of validation losses per epoch.
    """
    figure = plt.figure(figsize=(10, 6))
    title = "Learning Curve: Accuracy per Epoch"
    if model_name is not None:
        title += f" {model_name}"
    plt.title(title)
    plt.plot(epoch_list, train_acc, label="Train")
    plt.plot(epoch_list, validation_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    return figure


def plot_learning_curve(epoch_list, train_loss: list, validation_loss: list, model_name: str = None,
                        angle_train_loss=None, angle_valid_loss=None, range_train_loss=None, range_valid_loss=None):
    """
    Plot the learning curve.

    Args:
    -----
        epoch_list (list): List of epochs.
        train_loss (list): List of training losses per epoch.
        validation_loss (list): List of validation losses per epoch.
    """
    title = "Learning Curve: Loss per Epoch"
    if model_name is not None:
        title += f" {model_name}"
    if angle_train_loss is not None and range_train_loss is not None:

        # create 3 subplots, the main one will spread over 2 cols, and the other 2 will be under it.
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax_angle = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        ax_range = plt.subplot2grid((2, 2), (1, 1), colspan=1)

        ax.set_title(title)
        ax.plot(epoch_list, train_loss, label="Train")
        ax.plot(epoch_list, validation_loss, label="Validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        ax_angle.plot(epoch_list, angle_train_loss, label="Train")
        ax_angle.plot(epoch_list, angle_valid_loss, label="Validation")
        ax_angle.set_xlabel("Epochs")
        ax_angle.set_ylabel("Angle Loss [rad]")
        ax_angle.legend(loc="best")
        ax_range.plot(epoch_list, range_train_loss, label="Train")
        ax_range.plot(epoch_list, range_valid_loss, label="Validation")
        ax_range.set_xlabel("Epochs")
        ax_range.set_ylabel("Range Loss [m]")
        ax_range.legend(loc="best")
        # tight layout
        plt.tight_layout()
    else:
        fig = plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.plot(epoch_list, train_loss, label="Train")
        plt.plot(epoch_list, validation_loss, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="best")
    return fig


def simulation_summary(
        system_model_params: SystemModelParams,
        model_type: str,
        parameters: TrainingParams = None,
        phase="training",
):
    """
    Prints a summary of the simulation parameters.

    Parameters
    ----------
    system_model_params
    model_type
    parameters
    phase

    """
    M = system_model_params.M
    print("\n--- New Simulation ---\n")
    print(f"Description: Simulation of {model_type}, {phase} stage")
    print("System model parameters:")
    print(f"Number of sources = {M}")
    print(f"Number of sensors = {system_model_params.N}")
    print(f"field_type = {system_model_params.field_type}")
    print(f"signal_type = {system_model_params.signal_type}")
    print(f"Observations = {system_model_params.T}")
    print(
        f"SNR = {system_model_params.snr}, {system_model_params.signal_nature} sources"
    )
    print(f"Spacing deviation (eta) = {system_model_params.eta}")
    print(f"Bias spacing deviation (eta) = {system_model_params.bias}")
    print(f"Geometry noise variance = {system_model_params.sv_noise_var}")
    print("Simulation parameters:")
    print(f"Model: {model_type}")
    try:
        print(f"Model parameters: {parameters.model.get_model_params()}")
    except AttributeError:
        pass
    if phase.startswith("training"):
        print(f"Epochs = {parameters.epochs}")
        print(f"Batch Size = {parameters.batch_size}")
        print(f"Learning Rate = {parameters.learning_rate}")
        print(f"Weight decay = {parameters.weight_decay}")
        print(f"Gamma Value = {parameters.gamma}")
        print(f"Step Value = {parameters.step_size}")


def get_simulation_filename(
        system_model_params: SystemModelParams, model_config: ModelGenerator
):
    """

    Parameters
    ----------
    system_model_params
    model_config

    Returns
    -------
    File name to a simulation ran.
    """
    return (
        f"{model_config.model.get_model_name()}_"
        f"N={system_model_params.N}_"
        f"M={system_model_params.M}_"
        f"T={system_model_params.T}_"
        f"SNR_{system_model_params.snr}_"
        f"{system_model_params.signal_type}_"
        f"{system_model_params.field_type}_field_"
        f"{system_model_params.signal_nature}"
        f"_eta={system_model_params.eta}_"
        f"bias={system_model_params.bias}_"
        f"sv_noise={system_model_params.sv_noise_var}"
    )
