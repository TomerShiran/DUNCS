import numpy as np
from scipy import interpolate


class SteeringVectorGenerator:
    _instance = None  # shared across all calls

    def __new__(cls, array, dist_array_elems, params):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_instance(array, dist_array_elems, params)
        return cls._instance

    def _init_instance(self, array, dist_array_elems, params):
        self.array = array
        self.params = params
        self.dist_array_elems = dist_array_elems
        dist = dist_array_elems[params.signal_type]
        print("Initializing Steering Vector Generator with eta={}".format(self.params.eta))
        self._uniform_bias = np.random.uniform(-self.params.bias, self.params.bias, size=1)
        self._mis_distance = np.random.uniform(-self.params.eta * dist, self.params.eta * dist, size=self.params.N)


    def generate(self, theta, *, distance=None, f=1, nominal=False,
                 pattern_data=None, generate_search_grid=False):
        """Smart dispatch based on params.field_type and use of antenna pattern."""
        field_type = self.params.field_type.lower()

        if pattern_data is not None:
            return self._generate_antenna_pattern(theta, pattern_data)

        if field_type == "far":
            return self._generate_far_field(theta, f=f)
        elif field_type == "near":
            return self._generate_near_field(theta, distance=distance, f=f,
                                   nominal=nominal, generate_search_grid=generate_search_grid)
        else:
            raise ValueError(f"Unknown field type: {self.params.field_type}")

    def _generate_far_field(self, theta, f=1):
        f_sv = {"NarrowBand": 1, "Broadband": f}
        mis_geometry_noise = np.sqrt(self.params.sv_noise_var) * np.random.randn(self.params.N) if self.params.sv_noise_var else 0
        dist = self.dist_array_elems[self.params.signal_type]
        return (
            np.exp(
                -2j * np.pi * f_sv[self.params.signal_type]
                * (self._uniform_bias + self._mis_distance + dist)
                * self.array * np.sin(theta)
            )
            + mis_geometry_noise
        )

    def _generate_near_field(self, theta: np.ndarray, distance: np.ndarray, f: float = 1,
                             nominal=False, generate_search_grid: bool = False):
        f_sv = {"NarrowBand": 1, "Broadband": f}

        theta = np.atleast_1d(theta)[:, np.newaxis]
        distance = np.atleast_1d(distance)[:, np.newaxis]
        array = self.array[:, np.newaxis]
        array_square = np.power(array, 2)
        dist_array_elems = self.dist_array_elems[self.params.signal_type]
        dist_array_elems += self._mis_distance
        dist_array_elems = dist_array_elems[:, np.newaxis]

        first_order = np.einsum("nm, na -> na",
                                array,
                                np.tile(np.sin(theta), (1, self.params.N)).T * dist_array_elems)
        first_order = np.tile(first_order[:, :, np.newaxis], (1, 1, len(distance)))

        second_order = -0.5 * np.divide(np.power(np.outer(np.cos(theta), dist_array_elems), 2)[:, None, :],
                                        distance.T[:, :, None])
        second_order = np.einsum("nm, nkl -> nkl",
                                 array_square,
                                 np.transpose(second_order, (2, 0, 1)))

        time_delay = first_order + second_order

        if not generate_search_grid:
            time_delay = np.diagonal(time_delay, axis1=1, axis2=2)

        # need to divide here by the wavelength, seems that for the narrowband scenario,
        # wavelength = 1.
        if not nominal:
            # Calculate additional steering vector noise
            mis_geometry_noise = ((np.sqrt(2) / 2) * np.sqrt(self.params.sv_noise_var)
                                  * (np.random.randn(*time_delay.shape) + 1j * np.random.randn(*time_delay.shape)))
            return np.exp(2 * -1j * np.pi * time_delay) + mis_geometry_noise
        return np.exp(2 * -1j * np.pi * time_delay)

    @staticmethod
    def _generate_antenna_pattern(theta, antenna_pattern_data):
        theta_deg = np.rad2deg(theta)
        azimuth_base_array, phase_array, amps_array = antenna_pattern_data

        interp_phase = interpolate.interp1d(azimuth_base_array, phase_array, axis=0,
                                            bounds_error=False, fill_value="extrapolate")
        interp_amps = interpolate.interp1d(azimuth_base_array, amps_array, axis=0,
                                           bounds_error=False, fill_value="extrapolate")

        phase = np.deg2rad(interp_phase(theta_deg))
        amps = 10 ** (interp_amps(theta_deg) / 20)
        return amps * np.exp(1j * phase)

    @classmethod
    def reset_instance(cls):
        cls._instance = None
