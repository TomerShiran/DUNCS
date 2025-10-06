# Imports
import os
import time
import numpy as np
import torch.linalg
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from typing import List

# Internal imports
from src.utils import *
from src.metrics.criterions import RMSPELoss, ADMMObjective
from src.methods_pack.music import MUSIC
from src.methods_pack.esprit import ESPRIT
from src.models import SubspaceNet, get_model
from src.plotting import plot_spectrum
from src.system_model import SystemModel
from src.methods_pack.cov_reconstruct import CovReconstructor, get_cov_reconstruction_method, sample_covariance
from src.metrics import crb
from src.eval.reporting import normalize_result


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


def evaluate_dnn_model(model: nn.Module, dataset: DataLoader, mode: str = "test") -> dict:
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (DataLoader): The evaluation dataset.

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the evaluation loss is not implemented for the model type.
    """

    # Initialize values
    overall_loss_angle = 0.0
    overall_accuracy = None
    test_length = 0
    # Set model to eval mode
    model.eval()
    with (torch.no_grad()):
        for data in dataset:
            if mode == "valid":
                eval_loss = model.validation_step(data)
            else:
                eval_loss = model.test_step(data)

            if isinstance(eval_loss, tuple):
                eval_loss, acc = eval_loss
            else:
                acc = None

            overall_loss_angle += torch.sum(eval_loss).item()
            if acc is not None:
                if overall_accuracy is None:
                    overall_accuracy = 0.0
                overall_accuracy += acc
            if data[0].dim() == 2:
                test_length += 1
            else:
                test_length += data[0].shape[0]
            ############################################################################################################
    overall_loss_angle /= test_length
    if overall_accuracy is not None:
        overall_accuracy /= test_length
    return normalize_result((overall_loss_angle, overall_accuracy))


def evaluate_augmented_model(
        model: SubspaceNet,
        dataset,
        system_model,
        criterion=RMSPELoss,
        algorithm: str = "music",
        plot_spec: bool = False,
        figures: dict = None,
):
    """
    Evaluate an augmented model that combines a SubspaceNet model with another subspace method on a given dataset.

    Args:
    -----
        model (nn.Module): The trained SubspaceNet model.
        dataset: The evaluation dataset.
        system_model (SystemModel): The system model for the hybrid algorithm.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The hybrid algorithm to use (e.g., "music", "esprit"). Defaults to "music".
        plot_spec (bool): Whether to plot the spectrum for the hybrid algorithm. Defaults to False.
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
    --------
        float: The average evaluation loss.

    Raises:
    -------
        Exception: If the algorithm is not supported.
        Exception: If the algorithm is not supported
    """
    # Initialize parameters for evaluation
    hybrid_loss = []
    if not isinstance(model, SubspaceNet):
        raise Exception("evaluate_augmented_model: model is not from type SubspaceNet")
    # Set model to eval mode
    model.eval()
    # Initialize instances of subspace methods
    methods = {
        "music": MUSIC(system_model, estimation_parameter="angle"),
        "esprit": ESPRIT(system_model),
        "music_2D": MUSIC(system_model, estimation_parameter="angle, range")
    }
    # If algorithm is not in methods
    if methods.get(algorithm) is None:
        raise Exception(
            f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
        )
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            X, true_label = data
            if algorithm.endswith("2D"):
                DOA, RANGE = torch.split(true_label, true_label.size(1) // 2, dim=1)
                RANGE.to(device)
            else:
                DOA = true_label

            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Apply method with SubspaceNet augmentation
            method_output = methods[algorithm].narrowband(
                X=X, mode="SubspaceNet", model=model
            )
            # Calculate loss, if algorithm is "music" or "esprit"
            if not algorithm.startswith("mvdr"):
                if algorithm.endswith("2D"):
                    predictions_doa, predictions_distance = method_output[0], method_output[1]
                    loss = criterion(predictions_doa, DOA * R2D, predictions_distance, RANGE)
                else:
                    predictions, M = method_output[0], method_output[-1]
                    # If the amount of predictions is less than the amount of sources
                    predictions = add_random_predictions(M, predictions, algorithm)
                    # Calculate loss criterion
                    loss = criterion(predictions, DOA * R2D)
                hybrid_loss.append(loss)
            else:
                hybrid_loss.append(0)
            # Plot spectrum, if algorithm is "music"
            if not algorithm.startswith("esprit"):
                if plot_spec and i == len(dataset.dataset) - 1:
                    predictions, spectrum = method_output[0], method_output[1]
                    figures[algorithm]["norm factor"] = np.max(spectrum)
                    plot_spectrum(
                        predictions=predictions,
                        true_DOA=DOA * R2D,
                        system_model=system_model,
                        spectrum=spectrum,
                        algorithm="SubNet+" + algorithm.upper(),
                        figures=figures,
                    )
    return np.mean(hybrid_loss)


def evaluate_model_based(
        dataset: DataLoader,
        system_model: SystemModel,
        criterion: nn.Module,
        algorithm: str,
        cov_recon: CovReconstructor):
    """
    Evaluate different model-based algorithms on a given dataset.

    Args:
        dataset (DataLoader): The evaluation dataset.
        system_model (SystemModel): The system model for the algorithms.
        criterion (nn.Module): The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The algorithm to use (e.g., "music", "esprit", "r-music").
        cov_recon (CovReconstructor) : The method to use for the covariance matrix reconstruction

    Returns:
        overall_loss (Dict): Dict of evaluation loss and accuracy

    Raises:
        Exception: If the algorithm is not supported.
    """
    # Initialize parameters for evaluation
    overall_loss = 0.0
    overall_acc = 0.0
    test_length = 0

    model_based = get_model_based_method(algorithm, system_model)
    if isinstance(model_based, nn.Module):
        model_based = model_based.to(device)
        # Set model to eval mode
        model_based.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x, sources_num, angles = data
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(device)
            angles = angles.to(device)
            validate_constant_sources_number(sources_num)
            sources_num = sources_num[0]

            cov = cov_recon(x)
            angles_prediction, source_estimation, _ = model_based(cov, number_of_sources=sources_num)
            overall_loss += criterion(angles_prediction, angles).item()
            if source_estimation is not None:
                overall_acc += torch.sum(source_estimation == sources_num * torch.ones_like(source_estimation).float()).item()

            if data[0].dim() == 2:
                test_length += 1
            else:
                test_length += data[0].shape[0]

        overall_loss /= test_length
        overall_acc /= test_length
        return normalize_result((overall_loss, overall_acc))


def evaluate_admm_convergence(dataset: DataLoader,
                              criterion: nn.Module,
                              cov_recon: CovReconstructor):
    # Initialize parameters for evaluation
    overall_loss = 0.0
    test_length = 0

    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x, sources_num, _ = data
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(device)
            validate_constant_sources_number(sources_num)

            cov = cov_recon(x)
            Rx = sample_covariance(x)
            loss = criterion(cov, Rx)
            overall_loss += loss.item()
            if data[0].dim() == 2:
                test_length += 1
            else:
                test_length += data[0].shape[0]

        overall_loss /= test_length
        return normalize_result(overall_loss)


def add_random_predictions(M: int, predictions: np.ndarray, algorithm: str):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.
        algorithm (str): The algorithm used.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    while predictions.shape[0] < M:
        # print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(
            predictions, 0, np.round(np.random.rand(1) * 180, decimals=2) - 90.00
        )
    return predictions


def evaluate_crb(dataset: DataLoader,
                 system_model: SystemModel):
    params = system_model.params
    if system_model.is_sparse_array and params.field_type.lower() == "far":
        crb_sum_1 = 0.0
        test_length = 0
        for i, data in enumerate(dataset):
            x, sources_num, angles = data
            angles = angles.to(device)

            validate_constant_sources_number(sources_num)
            array_pos = torch.from_numpy(system_model.array).to(device=device, dtype=torch.long)

            crb_sncr_exact = crb.calculate_sncr_crb(
                sparse_array=array_pos,
                thetas_rad=angles,  # (B, K)
                snr_db=params.snr,
                L_snapshots=params.T,
                return_per_angle=False
            )
            if data[0].dim() == 2:
                test_length += 1
            else:
                test_length += data[0].shape[0]

            crb_sum_1 += torch.sum(crb_sncr_exact)

        crb1 = crb_sum_1 / test_length
        return normalize_result(crb1)

    else:
        print("CRB for this scenario isn't supported yet")
    return


def evaluate(
        generic_test_dataset: DataLoader,
        criterions: List[nn.Module],
        system_model: SystemModel,
        models: dict = None,
        augmented_methods: list = None,
        subspace_methods: list = None,
        model_tmp: nn.Module = None,
        cov_recon_method='sample',
        cov_recon_params: dict = None,
        admm_iterations: list = None):
    if cov_recon_method == 'admm':
        results = admm_evaluation(generic_test_dataset, criterions, system_model, model_tmp, subspace_methods,
                                  cov_recon_method, cov_recon_params, admm_iterations, models=models)
        # plot_admm_test_results(results)

    else:
        results = {}
        for crit in criterions:
            crit_name = crit.__class__.__name__
            # initialize per-criterion dict
            res = results.setdefault(crit_name, {})

            if model_tmp is not None:
                model_test_loss = evaluate_dnn_model(model_tmp, generic_test_dataset)
                model_name = model_tmp._get_name()
                res[model_name] = model_test_loss
            # Evaluate DNN models
            for model_name, params in models.items():
                model = get_model(model_name, params, system_model)
                num_of_params = sum(p.numel() for p in model.parameters())
                total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
                print(f"Number of parameters in {model_name}: {num_of_params} with total size: {total_size} bytes")
            #     start = time.time()
                model_test_loss = evaluate_dnn_model(model, generic_test_dataset)
            #     print(f"{model_name} evaluation time: {time.time() - start}")
                res[model_name] = model_test_loss

            # Evaluate classical subspace methods
            cov_recon = get_cov_reconstruction_method(cov_recon_method, system_model, **cov_recon_params)
            if isinstance(crit, ADMMObjective):
                loss = evaluate_admm_convergence(generic_test_dataset, crit, cov_recon)
                res[cov_recon.__class__.__name__] = loss
            else:
                for algorithm in subspace_methods:
                    start = time.time()
                    loss = evaluate_model_based(
                        generic_test_dataset,
                        system_model,
                        criterion=crit,
                        algorithm=algorithm,
                        cov_recon=cov_recon)
                    if system_model.params.signal_nature == "coherent" and algorithm.lower() in ["1d-music", "2d-music",
                                                                                                 "r-music", "esprit"]:
                        algorithm += "(SPS)"
                    print(f"{algorithm} evaluation time: {time.time() - start}")
                    res[algorithm] = loss
    # results[criterions[0].__class__.__name__]['crb_sncr'] = evaluate_crb(generic_test_dataset, system_model)

    for crit_name, method_dict in results.items():
        print(f"\n=== Results for {crit_name} ===")
        for method, loss in method_dict.items():
            print(f"{method} = {loss}")

    return results


def admm_evaluation(generic_test_dataset: DataLoader,
                    criterions: List[nn.Module],
                    system_model: SystemModel,
                    model: nn.Module = None,
                    subspace_methods: list = None,
                    cov_recon_method='admm',
                    cov_recon_params: dict = None,
                    admm_iterations: list = None,
                    models:dict = None):
    results = {}
    eval_models = [model] if model is not None else []
    if models:
        for model_name, params in models.items():
            if model_name == "DUNCS":
                eval_models.append(get_model(model_name, params, system_model))

    if admm_iterations is None:
        admm_iterations = [model.num_iter]
    for crit in criterions:
        crit_name = crit.__class__.__name__
        # initialize per-criterion dict
        res = results.setdefault(crit_name, {})
        print(f"\n=== Evaluating criterion {crit_name} ===")

        for eval_model in eval_models:
            eval_model.set_test_criteria(crit)

        for num_iterations in admm_iterations:
            print(f"\n=== Evaluating {num_iterations} Iterations ===")
            for eval_model in eval_models:
                if num_iterations <= eval_model.get_max_iterations():
                    # Evaluate DNN models if given
                    eval_model.set_num_test_iterations(num_iterations)
                    for subspace_method in subspace_methods:
                        eval_model.set_test_subspace_method(
                            subspace_method)  # TODO: Make it in a more robust and configurable way
                        model_test_loss = evaluate_dnn_model(eval_model, generic_test_dataset)
                        model_name = eval_model._get_name()
                        res[f"{model_name}_{eval_model.get_max_iterations()}_{subspace_method}_{num_iterations}"] = model_test_loss

            # Evaluate classical methods:
            cov_recon_params['max_iter'] = num_iterations
            cov_recon = get_cov_reconstruction_method(cov_recon_method, system_model, **cov_recon_params)
            if isinstance(crit, ADMMObjective):
                loss = evaluate_admm_convergence(generic_test_dataset, crit, cov_recon)
                res[f"{cov_recon.__class__.__name__}_{num_iterations}"] = loss
            else:
                for algorithm in subspace_methods:
                    loss = evaluate_model_based(
                        generic_test_dataset,
                        system_model,
                        criterion=crit,
                        algorithm=algorithm,
                        cov_recon=cov_recon)
                    if system_model.params.signal_nature == "coherent" and algorithm.lower() in ["1d-music", "2d-music",
                                                                                                 "r-music", "esprit"]:
                        algorithm += "(SPS)"
                    res[f"{algorithm}_{num_iterations}"] = loss

    return results
