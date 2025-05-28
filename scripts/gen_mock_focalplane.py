# -*- encoding: utf-8 -*-

"""
This script generates a mock focal plane for the LFT, MFT, and HFT telescopes.
The generated file `mock_focalplane.toml` is used for `test_pointing_sys.py`.

Usage:
    python gen_mock_focalplane.py
"""

from pathlib import Path

import numpy as np
import tomlkit

import litebird_sim as lbs


def gen_mock_detector(name, theta_rad, phi_rad):
    telescope, wafer, pix, orient_hand, freq, pol = name.split("_")
    if telescope == "000":
        telescope = "LFT"
        orient = orient_hand[0]
        # hand = orient_hand[1]
    elif telescope == "001":
        telescope = "MFT"
        orient = orient_hand[:2]
        # hand = orient_hand[-1]
        orient = str(orient)
    elif telescope == "002":
        telescope = "HFT"
        orient = orient_hand
    else:
        raise ValueError(f"Unknown telescope {telescope}")

    wafer = telescope[0] + wafer[-2:]

    data = {
        "name": name,
        "wafer": wafer,
        "pixel": int(pix),
        "pixtype": "def",
        "channel": "ghi",
        "bandcenter_ghz": 65.0,
        "bandwidth_ghz": 14.0,
        "sampling_rate_hz": 1.0,
        "fwhm_arcmin": 34.0,
        "ellipticity": 56.0,
        "net_ukrts": 78.0,
        "fknee_mhz": 90.0,
        "fmin_hz": 98.0,
        "alpha": 76.0,
        "pol": pol,
        "orient": orient,
        "quat": [0, 0, 0, 1],
    }

    det = lbs.DetectorInfo.from_dict(data)
    psi_rad = lbs.get_detector_orientation(det)

    q_psi = lbs.RotQuaternion(quats=np.array(lbs.quat_rotation_z(psi_rad)))
    q_theta = lbs.RotQuaternion(quats=np.array(lbs.quat_rotation_x(theta_rad)))
    q_phi = lbs.RotQuaternion(quats=np.array(lbs.quat_rotation_z(phi_rad)))

    quat = q_phi * q_theta * q_psi
    det.quat = quat
    data["quat"] = [quat.quats[0][i] for i in range(4)]
    return det, data


def append_to_toml_file_with_hierarchy(data, filename, section_name, sub_section_name):
    """
    A function to add a new hierarchical section and data to an existing TOML file.

    Parameters:
    - data (dict): The dictionary containing the data to be added.
    - filename (str): The name of the TOML file to which the data will be added.
    - section_name (str): The name of the parent section for the data to be added.
    - sub_section_name (str): The name of the child section for the data to be added.
    """
    # Load the file if it exists, otherwise create a new document
    try:
        with open(filename, "r") as file:
            doc = tomlkit.parse(file.read())
    except FileNotFoundError:
        doc = tomlkit.document()

    # Retrieve or create the parent section
    if section_name in doc:
        parent_section = doc[section_name]
    else:
        parent_section = tomlkit.table()
        doc[section_name] = parent_section

    # Create a new child section
    sub_section = tomlkit.table()
    for key, value in data.items():
        sub_section[key] = value

    # Add the child section to the parent section
    parent_section[sub_section_name] = sub_section

    # Save the modified document back to the file
    with open(filename, "w") as file:
        file.write(tomlkit.dumps(doc))


def save_to_toml(filename, telescope, orients, handiness, theta, phi):
    for i in range(len(orients)):
        if i % 2 == 0:
            pol = "T"
        else:
            pol = "B"
        if telescope == "LFT":
            name = f"000_000_00{i}_{orients[i]}{handiness[i]}_040_{pol}"
        elif telescope == "MFT":
            name = f"001_000_00{i}_{orients[i]}{handiness[i]}_100_{pol}"
        elif telescope == "HFT":
            name = f"002_000_00{i}_{orients[i]}_400_{pol}"
        det, data = gen_mock_detector(name, np.deg2rad(theta[i]), np.deg2rad(phi[i]))
        append_to_toml_file_with_hierarchy(data, filename, telescope, f"det_{i:03}")


filename = (
    Path(__file__).parent.parent
    / "test"
    / "pointing_sys_reference"
    / "mock_focalplane.toml"
)
telescope = "LFT"
orients = ["Q", "Q", "Q", "Q", "U", "U", "U", "U", "Q", "Q"]
handiness = ["A", "A", "A", "A", "B", "B", "B", "B", "A", "A"]
theta = [0, 0, 18, 18, 18, 18, 18, 18, 18, 18]
phi = [0, 0, 30, 30, 150, 150, 210, 210, 330, 330]
print(f"...generating mock {telescope} focal plane...")
save_to_toml(filename, telescope, orients, handiness, theta, phi)

telescope = "MFT"
fov = 14
sym = 60
orients = [
    "00",
    "00",
    "15",
    "15",
    "30",
    "30",
    "45",
    "45",
    "60",
    "60",
    "75",
    "75",
    "90",
    "90",
]
handiness = ["A", "A", "A", "A", "B", "B", "B", "B", "A", "A", "B", "B", "A", "A"]
theta = [0, 0, fov, fov, fov, fov, fov, fov, fov, fov, fov, fov, fov, fov]
phi = [
    0,
    0,
    sym,
    sym,
    2 * sym,
    2 * sym,
    3 * sym,
    3 * sym,
    4 * sym,
    4 * sym,
    5 * sym,
    5 * sym,
    6 * sym,
    6 * sym,
]
print(f"...generating mock {telescope} focal plane...")
save_to_toml(filename, telescope, orients, handiness, theta, phi)

telescope = "HFT"
sym = 120
orients = ["U", "U", "Q", "Q", "U", "U", "U", "U"]
theta = [0, 0, fov, fov, fov, fov, fov, fov]
phi = [0, 0, 0, 0, sym, sym, 2 * sym, 2 * sym]
print(f"...generating mock {telescope} focal plane...")
save_to_toml(filename, telescope, orients, handiness, theta, phi)
print("...done...")
