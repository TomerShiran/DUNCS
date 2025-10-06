"""
Subspace-Net

Details
----------
Name: plotting.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 29/06/23

Purpose
----------
This module provides functions for plotting subspace methods spectrums,
like RootMUSIC and MUSIC.
 
Functions:
----------

plot_spectrum(predictions: np.ndarray, true_DOA: np.ndarray, system_model=None,
    spectrum: np.ndarray =None, roots: np.ndarray =None, algorithm:str ="music",
    figures:dict = None): Wrapper spectrum plotter based on the algorithm.
plot_music_spectrum(system_model, figures: dict, spectrum: np.ndarray, algorithm: str):
    Plot the MUSIC spectrum.
plot_root_music_spectrum(roots: np.ndarray, predictions: np.ndarray,
    true_DOA: np.ndarray, algorithm: str): Plot the Root-MUSIC spectrum.
initialize_figures(void): Generates template dictionary containing figure objects for plotting multiple spectrums.


"""
# Imports
from datetime import datetime
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Dict

from src.methods import MUSIC
from src.utils import R2D
from src.utils import plot_styles, parse_loss_results_for_plotting


def plot_spectrum(predictions: np.ndarray, true_DOA: np.ndarray, system_model=None,
                  spectrum: np.ndarray = None, roots: np.ndarray = None, algorithm: str = "music",
                  figures: dict = None):
    """
  Wrapper spectrum plotter based on the algorithm.

  Args:
      predictions (np.ndarray): The predicted DOA values.
      true_DOA (np.ndarray): The true DOA values.
      system_model: The system model.
      spectrum (np.ndarray): The spectrum values.
      roots (np.ndarray): The roots for Root-MUSIC algorithm.
      algorithm (str): The algorithm used.
      figures (dict): Dictionary containing figure objects for plotting.

  Raises:
      Exception: If the algorithm is not supported.

  """
    # Convert predictions to 1D array
    if isinstance(predictions, (np.ndarray, list, torch.Tensor)):
        predictions = np.squeeze(np.array(predictions))
    # Plot MUSIC spectrums
    if "music" in algorithm.lower() and not ("r-music" in algorithm.lower()):
        plot_music_spectrum(system_model, figures, spectrum, algorithm)
    else:
        raise Exception(f"evaluate_augmented_model: Algorithm {algorithm} is not supported.")


def plot_music_spectrum(system_model, figures: dict, spectrum: np.ndarray, algorithm: str):
    """
    Plot the MUSIC spectrum.

    Args:
        system_model (SystemModel): The system model.
        figures (dict): Dictionary containing figure objects for plotting.
        spectrum (np.ndarray): The spectrum values.
        algorithm (str): The algorithm used.

    """
    # Initialize MUSIC instance
    music = MUSIC(system_model)
    angels_grid = music._angels * R2D
    # Initialize plot for spectrum
    if figures["music"]["fig"] == None:
        plt.style.use('default')
        figures["music"]["fig"] = plt.figure(figsize=(8, 6))
        # plt.style.use('plot_style.txt')
    if figures["music"]["ax"] == None:
        figures["music"]["ax"] = figures["music"]["fig"].add_subplot(111)
    # Set labels titles and limits
    figures["music"]["ax"].set_xlabel("Angels [deg]")
    figures["music"]["ax"].set_ylabel("Amplitude")
    figures["music"]["ax"].set_ylim([0.0, 1.01])
    # Apply normalization factor for multiple plots
    figures["music"]["norm factor"] = None
    if figures["music"]["norm factor"] != None:
        # Plot music spectrum
        figures["music"]["ax"].plot(angels_grid, spectrum / figures["music"]["norm factor"], label=algorithm)
    else:
        # Plot normalized music spectrum
        figures["music"]["ax"].plot(angels_grid, spectrum / np.max(spectrum), label=algorithm)
    # Set legend
    figures["music"]["ax"].legend()


def plot_root_music_spectrum(roots: np.ndarray, predictions: np.ndarray,
                             true_DOA: np.ndarray, algorithm: str):
    """
    Plot the Root-MUSIC spectrum.

    Args:
        roots (np.ndarray): The roots for Root-MUSIC polynomyal.
        predictions (np.ndarray): The predicted DOA values.
        true_DOA (np.ndarray): The true DOA values.
        algorithm (str): The algorithm used.

    """
    # Initialize figure
    plt.style.use('default')
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)
    # Set axis location and limits
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(90)
    ax.set_thetamax(-90)
    # plot roots ang angles 
    for i in range(len(predictions)):
        angle = predictions[i]
        r = np.abs(roots[i])
        ax.set_ylim([0, 1.2])
        ax.set_yticks([0, 1])
        ax.plot([0, angle * np.pi / 180], [0, r], marker='o')
    for doa in true_DOA:
        ax.plot([doa * np.pi / 180], [1], marker='x', color="r", markersize=14)
    ax.set_xlabel("Angels [deg]")
    ax.set_ylabel("Amplitude")
    plt.savefig("data/spectrums/{}_spectrum.pdf".format(algorithm), bbox_inches='tight')


def initialize_figures():
    """Generates template dictionary containing figure objects for plotting multiple spectrums.

  Returns:
      (dict): The figures dictionary
  """
    figures = {"music": {"fig": None, "ax": None, "norm factor": None},
               "r-music": {"fig": None, "ax": None},
               "esprit": {"fig": None, "ax": None}}
    return figures


def plot_results(loss_dict: dict, criterion: str = "RMSPE", plot_acc: bool = False, save_to_file: bool = False):
    """
    Plot the results of the simulation.
    The dict could be with several scenarios, each with different SNR values, or with different number of snapshots,
    or with diffetent noise to the steering matrix.

    Parameters
    ----------
    criterion
    loss_dict

    Returns
    -------

    """
    now = datetime.now()
    base_plot_path = Path(__file__).parent.parent / "data" / "simulations" / "results" / "plots"
    snr_plot_path = base_plot_path / "SNR"
    snapshots_plot_path = base_plot_path / "Snapshots"
    steering_noise_plot_path = base_plot_path / "SteeringNoise"
    base_plot_path.mkdir(parents=True, exist_ok=True)
    snr_plot_path.mkdir(parents=True, exist_ok=True)
    snapshots_plot_path.mkdir(parents=True, exist_ok=True)
    steering_noise_plot_path.mkdir(parents=True, exist_ok=True)

    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    plt.rcParams.update({'font.size': 18})
    for scenrio, dict_values in loss_dict.items():
        if scenrio == "SNR":
            plot_path = os.path.join(snr_plot_path, dt_string_for_save)
            plot_test_results(scenrio, dict_values, plot_path, criterion, save_to_file=save_to_file, plot_acc=plot_acc)
        elif scenrio == "T":
            plot_path = os.path.join(snapshots_plot_path, dt_string_for_save)
            plot_test_results(scenrio, dict_values, plot_path, criterion, save_to_file=save_to_file, plot_acc=plot_acc)
        elif scenrio == "eta":
            plot_path = os.path.join(steering_noise_plot_path, dt_string_for_save)
            plot_test_results(scenrio, dict_values, plot_path, criterion, save_to_file=save_to_file, plot_acc=plot_acc)
        else:
            raise ValueError(f"Unknown scenario: {scenrio}")

    return


def plot_test_results(test: str, res: dict, simulations_path: str, criterion: str,
                      save_to_file=False, plot_acc: bool=False):
    """
    """
    # The input dict is a nested dict - the first level is for the snr values, the second level is for the methods,
    # and the third level is for the loss values or accuracy.
    # For example: res = {10: {"MUSIC": {"Overall": 0.1, "Accuracy": 0.9}, "RootMUSIC": {"Overall": 0.2, "Accuracy": 0.8}}
    # Or, for near filed scenrio: res = {10: {"MUSIC": {"Overall": 0.1, "Angle": 0.2, "Distance": 0.3, "Accuracy": 0.9},
    # "RootMUSIC": {"Overall": 0.2, "Angle": 0.3, "Distance": 0.4, "Accuracy": 0.8}}
    # The possible test are: "SNR", "T", "eta"

    # create a plot based on the criterion and the test type
    if criterion == "rmspe":
        if False:
        # if not None in np.stack([list(d.values()) for d in list(next(iter(res.values())).values())]):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
            test_values = res.keys()
            if test == "eta":
                test_values = np.array(list(res.keys())) * 2
            plt_res, plt_acc = parse_loss_results_for_plotting(res)
            for method, loss_ in plt_res.items():
                if loss_.get("Accuracy") is not None and method != "TransMUSIC" and test == "SNR":
                    label = method + f": {np.mean(loss_['Accuracy']) * 100:.2f} %"
                else:
                    label = method
                ax1.plot(test_values, loss_["Angle"], label=label, **plot_styles[method.split("_")[0]])
                ax2.plot(test_values, loss_["Distance"], label=label, **plot_styles[method.split("_")[0]])
            ax1.legend()
            ax2.legend()
            ax1.grid()
            ax2.grid()
            if test == "SNR":
                ax1.set_xlabel("SNR [dB]")
                ax2.set_xlabel("SNR [dB]")
            elif test == "T":
                ax1.set_xlabel("T")
                ax2.set_xlabel("T")
            elif test == "eta":
                ax1.set_xlabel("$\eta[{\lambda}/{2}]$")
                ax2.set_xlabel("$\eta[{\lambda}/{2}]$")
            ax1.set_ylabel("RMSPE [rad]")
            ax2.set_ylabel("RMSPE [m]")
            # ax1.set_title("Angle RMSE")
            # ax2.set_title("Distance RMSE")
            ax1.set_yscale("log")
            ax2.set_yscale("log")
            if save_to_file:
                fig.savefig(simulations_path + "_loss.pdf", transparent=True, bbox_inches='tight')
            fig.show()
            if plt_acc and plot_acc:
                plot_acc_results(test, test_values, plt_res, simulations_path, save_to_file)

        else:  # FAR
            plot_overall_rmse(test, res, simulations_path, save_to_file, plot_acc=plot_acc)

    elif criterion == "cartesian":
        plot_overall_rmse(test, res, simulations_path, save_to_file, units="m", plot_acc=plot_acc)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def plot_overall_rmse(test: str, res: dict, simulations_path: str,
                      save_to_file=False, units="rad", plot_acc: bool=False):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    test_values = res.keys()
    if test == "eta":
        test_values = np.array(list(res.keys())) * 2
    plt_res, plt_acc = parse_loss_results_for_plotting(res)
    for method, loss_ in plt_res.items():
        if loss_.get("Accuracy") is not None and method != "TransMUSIC" and test == "SNR":
            label = method + f": {np.mean(loss_['Accuracy']) * 100:.2f} %"
        else:
            label = method
        if not np.isnan((loss_.get("Overall"))).any():
            ax.plot(test_values, loss_["Overall"], **plot_styles[method.split("_")[0]], label=label)
    ax.legend()
    ax.grid()
    if test == "SNR":
        ax.set_xlabel("SNR [dB]")
    elif test == "T":
        ax.set_xlabel("T")
    elif test == "eta":
        ax.set_xlabel("$\eta[{\lambda}/{2}]$")
    ax.set_ylabel(f"RMSPE [{units}]")
    # ax.set_title("Overall RMSPE loss")
    ax.set_yscale("linear")
    ax.set_xticks(list(test_values))
    if save_to_file:
        fig.savefig(simulations_path + "_loss.pdf", transparent=True, bbox_inches='tight')
    fig.show()
    if plt_acc and plot_acc:
        plot_acc_results(test, test_values, plt_res, simulations_path, save_to_file)


def plot_acc_results(test, test_values, plt_res, simulations_path, save_to_file=False):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for method, loss_ in plt_res.items():
        if loss_.get("Accuracy") is not None:
            ax.plot(test_values, loss_["Accuracy"], label=method, **plot_styles[method.split("_")[0]])
    ax.legend()
    ax.grid()
    if test == "SNR":
        ax.set_xlabel("SNR [dB]")
    elif test == "T":
        ax.set_xlabel("T")
    elif test == "eta":
        ax.set_xlabel("eta")
    ax.set_ylabel("Accuracy [%]")
    # ax.set_title("Accuracy")
    ax.set_yscale("linear")
    if save_to_file:
        fig.savefig(simulations_path + "_acc.pdf", transparent=True, bbox_inches='tight')
    fig.show()


def plot_admm_test_results(results: Dict[str, Dict[str, float]]):
    """
    Plot loss vs. number of iterations for each method/DNN line,
    with a separate plot for each criterion.

    Parameters:
        results: {
            'RMSPELoss': {
                'ModelA_1': 0.123,
                'ADMM_1'  : 0.456,
                'Music_1' : 0.789,
                'ModelA_5': 0.111,
                'ADMM_5'  : 0.222,
                'Music_5' : 0.333,
                ...
            },
            'CustomLoss': { ... },
        }
    """
    for crit_name, method_dict in results.items():
        # Aggregate loss values per method
        method_data = {}
        for key, loss in method_dict.items():
            if '_' not in key:
                continue
            method_name, iter_str = key.rsplit('_', 1)
            try:
                iterations = int(iter_str)
            except ValueError:
                # skip keys that don't follow the expected naming
                continue
            method_data.setdefault(method_name, []).append((iterations, loss))

        # Create a new figure for each criterion
        plt.figure()
        for method_name, points in method_data.items():
            # Sort by iteration count
            points_sorted = sorted(points, key=lambda x: x[0])
            xs = [pt[0] for pt in points_sorted]
            ys = [pt[1] for pt in points_sorted]

            if crit_name == 'ADMMObjective':
                ys = np.log(ys)
            plt.plot(xs, ys, label=method_name)

        plt.xlabel('Number of ADMM Iterations')
        ylabel = 'Log(Loss)' if crit_name == 'ADMMObjective' else 'Loss'
        plt.ylabel(ylabel)
        plt.title(f'{crit_name} vs Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()