"""Subspace-Net 
Details
----------
Name: system_model.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the SystemModel class for defining the settings of the DoA estimation system model.
"""

# Imports
import numpy as np

from typing import Optional

from src.steering_vector_generator import SteeringVectorGenerator
from src.sparse_array import get_array_locations, get_virtual_ula_array, get_difference_co_array
from src.config.simulation_config import SystemModelParams


class SystemModel(object):
    def __init__(self, system_model_params: SystemModelParams):
        """Class used for defining the settings of the system model.

        Attributes:
        -----------
            field_type (str): Field environment approximation type. Options: "Far", "Near".
            signal_type (str): Signals type. Options: "NarrowBand", "Broadband".
            N (int): Number of sensors.
            M (int): Number of sources.
            freq_values (list, optional): Frequency range for broadband signals. Defaults to None.
            min_freq (dict): Minimal frequency value for different scenarios.
            max_freq (dict): Maximal frequency value for different scenarios.
            f_rng (dict): Frequency range of interest for different scenarios.
            f_sampling (dict): Sampling rate for different scenarios.
            time_axis (dict): Time axis for different scenarios.
            dist (dict): Distance between array elements for different scenarios.
            array (np.ndarray): Array of sensor locations.

        Methods:
        --------
            define_scenario_params(freq_values: list): Defines the signal_type parameters.
            create_array(): Creates the array of sensor locations.
            steering_vec(theta: np.ndarray, f: float = 1, array_form: str = "ULA",
                eta: float = 0, geo_noise_var: float = 0) -> np.ndarray: Computes the steering vector.

        """
        self.array = None
        self.virtual_array_ula_seg = None

        self.dist_array_elems = None
        self.time_axis = None
        self.f_sampling = None
        self.max_freq = None
        self.min_freq = None
        self.f_rng = None
        self.is_sparse_array = False if system_model_params.array_form.lower() == 'ula' else True
        self.params = system_model_params
        # Assign signal type parameters
        self.define_scenario_params()

        # # Define array indices
        self.create_array(system_model_params.array_form)
        # Calculation for the Fraunhofer and Fresnel
        self.fraunhofer, self.fresnel = self.calc_fresnel_fraunhofer_distance()

        self.sv_generator = SteeringVectorGenerator(
            array=self.array,
            dist_array_elems=self.dist_array_elems,
            params=self.params
        )


    def define_scenario_params(self):
        """Defines the signal type parameters based on the specified frequency values."""
        freq_values = self.params.freq_values
        # Define minimal frequency value
        self.min_freq = {"NarrowBand": None, "Broadband": freq_values[0]}
        # Define maximal frequency value
        self.max_freq = {"NarrowBand": None, "Broadband": freq_values[1]}
        # Frequency range of interest
        self.f_rng = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                start=self.min_freq["Broadband"],
                stop=self.max_freq["Broadband"],
                num=self.max_freq["Broadband"] - self.min_freq["Broadband"],
                endpoint=False,
            ),
        }
        # Define sampling rate as twice the maximal frequency
        self.f_sampling = {
            "NarrowBand": None,
            "Broadband": 2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"]),
        }
        # Define time axis
        self.time_axis = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                0, 1, self.f_sampling["Broadband"], endpoint=False
            ),
        }
        # distance between array elements
        self.dist_array_elems = {
            "NarrowBand": 1 / 2,
            "Broadband": 1
                         / (2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"])),
        }

    def create_array(self, array_form: str):
        """create an array of sensors locations, around to origin."""
        if array_form.lower() == 'ula':
            self.array = np.linspace(0, self.params.N, self.params.N, endpoint=False)
        elif self.is_sparse_array:
            self.array = get_array_locations(array_form)
            self.virtual_array_ula_seg = get_virtual_ula_array(self.array)
        else:
            raise ValueError(f"{array_form} isn't supported")

    def calc_fresnel_fraunhofer_distance(self) -> tuple:
        """
        In the Far and Near field scenrios, those distances are relevant for the distance grid creation.
        wavelength = 1
        spacing = wavelength / 2
        diemeter = (N-1) * spacing
        Fraunhofer  = 2 * diemeter ** 2 / wavelength
        Fresnel = 0.62 * (diemeter ** 3 / wavelength) ** 0.5
        Returns:
            tuple: fraunhofer(float), fresnel(float)
        """
        wavelength = 1
        spacing = wavelength / 2
        diemeter = (self.params.N - 1) * spacing
        fraunhofer = 2 * diemeter ** 2 / wavelength
        fresnel = 0.62 * (diemeter ** 3 / wavelength) ** 0.5
        # fresnel = ((diemeter ** 4) / (8 * wavelength)) ** (1 / 3)

        return fraunhofer, fresnel

    def steering_vec(self, theta, *, distance: Optional[np.ndarray] = None,
                     f: float = 1, nominal=False,
                     pattern_data=None, generate_search_grid=False):
        return self.sv_generator.generate(
            theta,
            distance=distance,
            f=f,
            nominal=nominal,
            pattern_data=pattern_data,
            generate_search_grid=generate_search_grid
        )