"""Subspace-Net 
Details
----------
Name: signal_creation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the Samples class, which inherits from SystemModel class.
This class is used for defining the samples model.
"""

# Imports
import numpy as np
from src.system_model import SystemModel
from src.utils import D2R, resolve_param
from src.config.simulation_config import SystemModelParams

class Samples(SystemModel):
    """
    Class used for defining and creating signals and observations.
    Inherits from SystemModel class.

    ...

    Attributes:
    -----------
        doa (np.ndarray): Array of angels (directions) of arrival.

    Methods:
    --------
        set_doa(doa): Sets the direction of arrival (DOA) for the signals.
        samples_creation(noise_mean: float = 0, noise_variance: float = 1, signal_mean: float = 0,
            signal_variance: float = 1): Creates samples based on the specified mode and parameters.
        noise_creation(noise_mean, noise_variance): Creates noise based on the specified mean and variance.
        signal_creation(source_number, signal_mean=0, signal_variance=1): Creates signals based on the specified mode and parameters.
    """

    def __init__(self, system_model_params: SystemModelParams, use_real_antenna_pattern: bool = False):
        """Initializes a Samples object.

        Args:
        -----
        system_model_params (SystemModelParams): an instance of SystemModelParams,
            containing all relevant system model parameters.

        """
        super().__init__(system_model_params)
        self.distances = None

    def set_doa(self, doa, M):
        """
        Sets the direction of arrival (DOA) for the signals.

        Args:
        -----
            doa (np.ndarray): Array containing the DOA values.

        """

        def create_doa_with_gap(M: int):
            """
            Create M DOA values in the given range (in degrees) such that the difference
            between consecutive values is at least 'gap' (in degrees).

            The method first reserves (M-1)*gap of the total range, then distributes the
            remaining extra space randomly among the points.

            Args:
                gap (float): Minimal gap (in degrees) between consecutive DOAs.
                M (int): Number of DOA values to generate.
                doa_range (tuple): A tuple (L, U) specifying the range in degrees.

            Returns:
                np.ndarray: An array of DOA values (sorted in ascending order) that satisfy the minimal gap.

            Raises:
                ValueError: If the range is too small for M points with the given gap.
            """
            L, U = self.params.doa_range
            total_range = U - L

            if total_range < (M - 1) * self.params.min_gap:
                raise ValueError("Invalid parameters. Use a smaller gap or a larger DOA range.")

            # The remaining space after reserving the minimum gap
            extra = total_range - (M - 1) * self.params.min_gap

            # Generate M random numbers in [0, 1] and sort them.
            r = np.sort(np.random.rand(M))

            # Compute the DOAs: fixed gap increments plus a random extra offset.
            DOA = L + self.params.min_gap * np.arange(M) + extra * r

            return DOA

        if doa == None:
            # Generate angels with gap greater than 0.2 rad (nominal case)
            self.doa = np.array(create_doa_with_gap(M=M)) * D2R
        else:
            # Generate
            self.doa = np.array(doa) * D2R

    def set_range(self, distance: list | np.ndarray, M) -> np.ndarray:
        """

        Args:
            distance:

        Returns:

        """

        def choose_distances(M, distance_min_gap: float = 0.5, distance_max_gap: int = 10,
                             min_val: float = 2, max_val: int = 7) -> np.ndarray:

            distances = np.round(np.random.uniform(min_val, max_val, M), decimals=0)  # TODO
            if np.unique(distances).shape[0] != M:
                distances = np.round(np.random.uniform(min_val, max_val, M), decimals=0)
            # distances = np.zeros(M)
            # idx = 0
            # while idx < M:
            #     distance = np.round(np.random.uniform(min_val, max_val), decimals=0)
            #     if len(distances) == 0:
            #         distances[idx] = distance
            #         idx += 1
            #     else:
            #         if np.min(np.abs(np.array(distances) - distance)) >= distance_min_gap and \
            #                 np.max(np.abs(np.array(distances) - distance)) <= distance_max_gap:
            #             distances[idx] = distance
            #             idx += 1
            return distances

        if distance is None:
            self.distances = choose_distances(M, min_val=self.fresnel, max_val=self.fraunhofer * 0.4,
                                              distance_min_gap=0.5, distance_max_gap=self.fraunhofer)
        else:
            self.distances = distance

    def samples_creation(
        self,
        noise_mean: float = 0,
        noise_variance: float = 1,
        signal_mean: float = 0,
        signal_variance: float = 1,
        source_number: int = None,
    ):
        """Creates samples based on the specified mode and parameters.

        Args:
        -----
            noise_mean (float, optional): Mean of the noise. Defaults to 0.
            noise_variance (float, optional): Variance of the noise. Defaults to 1.
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            tuple: Tuple containing the created samples, signal, steering vectors, and noise.

        Raises:
        -------
            Exception: If the signal_type is not defined.

        """
        # Generate signal matrix
        signal = self.signal_creation(source_number, signal_mean, signal_variance)
        # Generate noise matrix
        noise = self.noise_creation(noise_mean, noise_variance)
        # Generate Narrowband samples
        if self.params.signal_type.startswith("NarrowBand"):
            if self.params.field_type.startswith("Far"):
                A = np.array([self.steering_vec(theta) for theta in self.doa]).T
                clear_obs = A @ signal
            elif self.params.field_type.startswith("Near"):
                A = self.steering_vec(theta=self.doa, distance=self.distances, nominal=False, generate_search_grid=False)
                clear_obs = A @ signal
            else:
                raise Exception(f"Samples.params.field_type: Field type {self.params.field_type} is not defined")
            return clear_obs, noise
        # Generate Broadband samples
        elif self.params.signal_type.startswith("Broadband"):
            samples = []
            SV = []

            for idx in range(self.f_sampling["Broadband"]):
                # mapping from index i to frequency f
                if idx > int(self.f_sampling["Broadband"]) // 2:
                    f = -int(self.f_sampling["Broadband"]) + idx
                else:
                    f = idx
                A = np.array([self.steering_vec(theta, f) for theta in self.doa]).T
                samples.append((A @ signal[:, idx]) + noise[:, idx])
                SV.append(A)
            samples = np.array(samples)
            SV = np.array(SV)
            samples_time_domain = np.fft.ifft(samples.T, axis=1)[:, : self.params.T]
            return samples_time_domain, signal, SV, noise
        else:
            raise Exception(
                f"Samples.samples_creation: signal type {self.params.signal_type} is not defined"
            )

    def noise_creation(self, noise_mean, noise_variance):
        """Creates noise based on the specified mean and variance.

        Args:
        -----
            noise_mean (float): Mean of the noise.
            noise_variance (float): Variance of the noise.

        Returns:
        --------
            np.ndarray: Generated noise.

        """
        # for NarrowBand signal_type Noise represented in the time domain
        if self.params.signal_type.startswith("NarrowBand"):
            return (
                np.sqrt(noise_variance)
                * (np.sqrt(2) / 2)
                * (
                    np.random.randn(self.params.N, self.params.T)
                    + 1j * np.random.randn(self.params.N, self.params.T)
                )
                + noise_mean
            )
        # for Broadband signal_type Noise represented in the frequency domain
        elif self.params.signal_type.startswith("Broadband"):
            noise = (
                np.sqrt(noise_variance)
                * (np.sqrt(2) / 2)
                * (
                    np.random.randn(self.params.N, len(self.time_axis["Broadband"]))
                    + 1j
                    * np.random.randn(self.params.N, len(self.time_axis["Broadband"]))
                )
                + noise_mean
            )
            return np.fft.fft(noise)
        else:
            raise Exception(
                f"Samples.noise_creation: signal type {self.params.signal_type} is not defined"
            )

    def signal_creation(self, source_number: int, signal_mean: float = 0, signal_variance: float = 1):
        """
        Creates signals based on the specified signal nature and parameters.

        Args:
        -----
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            np.ndarray: Created signals.

        Raises:
        -------
            Exception: If the signal type is not defined.
            Exception: If the signal nature is not defined.
        """
        M = source_number
        # amplitude = 10 ** (resolve_param(self.params.snr) / 20)
        # NarrowBand signal creation
        if self.params.signal_type == "NarrowBand":
            if self.params.signal_nature == "non-coherent":
                # create M non-coherent signals
                return (
                    (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(M, self.params.T)
                        + 1j * np.random.randn(M, self.params.T)
                    )
                    + signal_mean
                )

            elif self.params.signal_nature == "coherent":
                # Coherent signals: same amplitude and phase for all signals
                sig = (
                    (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(1, self.params.T)
                        + 1j * np.random.randn(1, self.params.T)
                    )
                    + signal_mean
                )
                return np.repeat(sig, M, axis=0)

        # OFDM Broadband signal creation
        elif self.params.signal_type.startswith("Broadband"):
            num_sub_carriers = self.max_freq[
                "Broadband"
            ]  # number of subcarriers per signal
            if self.params.signal_nature == "non-coherent":
                # create M non-coherent signals
                signal = np.zeros(
                    (M, len(self.time_axis["Broadband"]))
                ) + 1j * np.zeros((M, len(self.time_axis["Broadband"])))
                for i in range(M):
                    for j in range(num_sub_carriers):
                        sig_amp = (
                            (np.sqrt(2) / 2)
                            * (np.random.randn(1) + 1j * np.random.randn(1))
                        )
                        signal[i] += sig_amp * np.exp(
                            1j
                            * 2
                            * np.pi
                            * j
                            * len(self.f_rng["Broadband"])
                            * self.time_axis["Broadband"]
                            / num_sub_carriers
                        )
                    signal[i] *= 1 / num_sub_carriers
                return np.fft.fft(signal)
            # Coherent signals: same amplitude and phase for all signals
            elif self.params.signal_nature == "coherent":
                signal = np.zeros(
                    (1, len(self.time_axis["Broadband"]))
                ) + 1j * np.zeros((1, len(self.time_axis["Broadband"])))
                for j in range(num_sub_carriers):
                    sig_amp = (
                        (np.sqrt(2) / 2)
                        * (np.random.randn(1) + 1j * np.random.randn(1))
                    )
                    signal += sig_amp * np.exp(
                        1j
                        * 2
                        * np.pi
                        * j
                        * len(self.f_rng["Broadband"])
                        * self.time_axis["Broadband"]
                        / num_sub_carriers
                    )
                signal *= 1 / num_sub_carriers
                return np.tile(np.fft.fft(signal), (M, 1))
            else:
                raise Exception(
                    f"signal nature {self.params.signal_nature} is not defined"
                )

        else:
            raise Exception(f"signal type {self.params.signal_type} is not defined")
