import numpy as np


def get_array_locations(array_form: str):
    array_type = array_form.lower()
    if array_type.startswith("mra"):
        if "4" in array_type:
            return np.array([0, 1, 4, 6])
        if "5" in array_type:
            return np.array([0, 1, 4, 7, 9])
        elif "6" in array_type:
            return np.array([0, 1, 6, 9, 11, 13])
        elif "7" in array_type:
            return np.array([0, 1, 4, 10, 12, 15, 17])
        elif "8" in array_type:
            return np.array([0, 1, 4, 10, 16, 18, 21, 23])
        else:
            raise Exception(f"{array_type} isn't supported")
    else:
        raise Exception(f"{array_type} isn't supported")


def get_difference_co_array(array_loc):
    array_locations = np.array(array_loc, dtype=float)  # Ensure numpy array
    coarray = array_locations[:, None] - array_locations[None, :]  # Difference coarray
    unique_lags = np.sort(np.unique(coarray))
    return unique_lags[unique_lags > 0]


def get_virtual_ula_array(array_loc):
    unique_lags = get_difference_co_array(array_loc)
    largest_ula_element = 0

    for i in range(1, len(unique_lags) + 1):
        if i not in unique_lags:
            break
        largest_ula_element = i

    return np.arange(0, largest_ula_element + 1, 1)




