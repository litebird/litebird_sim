# -*- encoding: utf-8 -*-

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
from pathlib import Path
import toml
from rich import print
from .quaternions import (
    quat_left_multiply,
    quat_right_multiply,
    quat_rotation_z,
)
from .simulations import Simulation
from .imo import Imo
from .detectors import DetectorInfo, FreqChannelInfo

CONFIG_FILE_PATH = Path.home() / ".config" / "litebird_imo" / "imo.toml"


class DetectorInfoViewer:
    def __init__(self):
        self.selected_detector_list = []
        self.scatter = []
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.info_box = None
        self.x_tot = None
        self.y_tot = None
        self.x_ch = None
        self.y_ch = None
        self.ndets_in_channel = None
        self.channel_dets_list = []
        self.total_dets_list = []
        self.base_path = None
        self.telescope = None
        self.channel = None
        self.imo_version = None

    def get_det_xy(self, det_info, telescope):
        """Get the x, y coordinates of the detector on the focal plane.

        Args:
            det_info (DetectorInfo): Detector information.

        Returns:
            x (float): The x coordinate of the detector on the focal plane.
            y (float): The y coordinate of the detector on the focal plane.
        """
        q = det_info.quat.quats[0]
        r = np.array([0.0, 0.0, 1.0, 0.0])
        if telescope == "LFT":
            # Rotate the FPU 180 degrees for projection if LFT is selected
            q_z = quat_rotation_z(np.deg2rad(180))
            for q_val in [q, q_z]:
                q_conj = np.array([-q_val[0], -q_val[1], -q_val[2], q_val[3]])
                quat_left_multiply(r, *q_val)
                quat_right_multiply(r, *q_conj)
        else:
            q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
            quat_left_multiply(r, *q)
            quat_right_multiply(r, *q_conj)
        theta, phi = hp.vec2ang(r[:3])
        x = np.rad2deg(theta) * np.cos(phi)
        y = np.rad2deg(theta) * np.sin(phi)
        return x, y

    def gen_info_text(self, detector):
        """Generate the information text of the detector.

        Args:
            detector (DetectorInfo): Detector information.
        """
        info_text = rf"""
        Detector info.
          $\cdot~$name: {detector.name}
	      $\cdot~$bandcenter_ghz: {detector.bandcenter_ghz}
	      $\cdot~$bandwidth_ghz: {detector.bandcenter_ghz}
	      $\cdot~$quat: {detector.quat.quats[0]}
	      $\cdot~$orient: {detector.orient}
	      $\cdot~$wafer: {detector.wafer}
	      $\cdot~$pixel: {detector.pixel}
	      $\cdot~$channel: {detector.channel}
	      $\cdot~$sampling_rate_hz: {detector.sampling_rate_hz}
	      $\cdot~$fwhm_arcmin: {detector.fwhm_arcmin}
	      $\cdot~$ellipticity: {detector.ellipticity}
	      $\cdot~$net_ukrts: {detector.net_ukrts}
	      $\cdot~$pol_sensitivity_ukarcmin: {detector.pol_sensitivity_ukarcmin}
	      $\cdot~$fknee_mhz: {detector.fknee_mhz}
	      $\cdot~$fmin_hz: {detector.fmin_hz}
	      $\cdot~$ellipticity: {detector.ellipticity}
	      $\cdot~$pointing_u_v: {detector.pointing_u_v}
	      $\cdot~$pointing_theta_phi_psi_deg: {detector.pointing_theta_phi_psi_deg}
	      $\cdot~$pol_angle_rad: {detector.pol_angle_rad}
	      $\cdot~$mueller_hwp: {detector.mueller_hwp}
        """
        return info_text

    def gen_detsinfo_text(self, det1, det2):
        """Generate the information text of the detector.

        Args:
            detector (DetectorInfo): Detector information.
        """
        info_text = rf"""
		Detector info.
		    name: {det1.name}, {det2.name}
		    bandcenter_ghz: {det1.bandcenter_ghz}
		    bandwidth_ghz: {det1.bandcenter_ghz}
		    quat: {det1.quat.quats[0]}
		           : {det2.quat.quats[0]}
		    orient: {det1.orient}
		    wafer: {det1.wafer}
		    pixel: {det1.pixel}
		    channel: {det1.channel}
		    sampling_rate_hz: {det1.sampling_rate_hz}
		    fwhm_arcmin: {det1.fwhm_arcmin}
		    ellipticity: {det1.ellipticity}
		    net_ukrts: {det1.net_ukrts}
		    pol_sensitivity_ukarcmin: {det1.pol_sensitivity_ukarcmin}
		    fknee_mhz: {det1.fknee_mhz}
		    fmin_hz: {det1.fmin_hz}
		    ellipticity: {det1.ellipticity}
		    pointing_u_v: {det1.pointing_u_v}
		    pointing_theta_phi_psi_deg: {det1.pointing_theta_phi_psi_deg}
		    pol_angle_rad: {det1.pol_angle_rad}, {det2.pol_angle_rad}
		    mueller_hwp: {det1.mueller_hwp}
        """
        return info_text

    def generate_dets_list_file(
        self, filename, selected_detector_list, duration_yr=1.0
    ):
        """Generate a text file with the selected detector list.
        The NET on detector is scaled by the`scaling_factor`:
        :math:`\\sqrt{\\frac{duration_yr}{3} \\times \\frac{N_{\\rm dets}^{\\rm e2e}}{N_{\\rm dets}^{\\rm ch}}}`

        Args:
            filename (str): The name of the file to be created.
            selected_detector_list (list): A list of selected detectors.
            duration_yr (float): The duration of the end-to-end simulation in years.
        """
        header = "# Telescope     Channel         IMO_NET         Number_det      Scaled_NET      Detector_name\n"
        selected_number_of_dets = len(selected_detector_list)
        scaling_factor = np.sqrt(
            duration_yr / 3.0 * selected_number_of_dets / self.ndets_in_channel
        )
        with open(os.path.join(self.base_path, filename), "w") as file:
            file.write(header)
            for detector in selected_detector_list:
                scaled_net = np.round(detector.net_ukrts * scaling_factor, 2)
                line = f"'LMHFT'\t\t{self.channel}\t\t{detector.net_ukrts}\t\t{selected_number_of_dets}/{self.ndets_in_channel}\t\t{scaled_net}\t\t{detector.name}\n"
                file.write(line)
        print(f"[green]The {filename} is generated.[/green]")
        print(f"[green]Location:[/green] [cyan]{self.base_path}/{filename}[/cyan]")

    def on_plot_click(self, event):
        """Select the detector by clicking on the plot.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The mouse event.
        """
        if not event.inaxes:
            return
        else:
            blue = "#1f77b4"
            red = "#b41f44"
            # distance between the clicked point and the detector
            distance = (
                (self.x_ch - event.xdata) ** 2 + (self.y_ch - event.ydata) ** 2
            ) ** 0.5
            if np.min(distance) < 0.3:
                sorted_indices = np.argsort(distance)
                indices = [sorted_indices[0], sorted_indices[1]]
                dets_list = []
                for idx in indices:
                    detector = self.channel_dets_list[idx]
                    dets_list.append(detector)
                    if self.scatter[idx].get_markerfacecolor() == blue:
                        self.scatter[idx].set_markerfacecolor(red)
                        self.scatter[idx].set_markeredgecolor(red)
                        self.scatter[idx].set_marker("*")
                        self.scatter[idx].set_markersize(12)
                        self.selected_detector_list.append(detector)
                    elif self.scatter[idx].get_markerfacecolor() == red:
                        self.scatter[idx].set_markerfacecolor(blue)
                        self.scatter[idx].set_markeredgecolor(blue)
                        self.scatter[idx].set_marker("o")
                        self.scatter[idx].set_markersize(8)
                        if detector in self.selected_detector_list:
                            self.selected_detector_list.remove(detector)
                info_text = self.gen_detsinfo_text(dets_list[0], dets_list[1])
                self.info_box.set_text(info_text)
                self.fig.canvas.draw()

    def ask_yes_or_no(self):
        while True:
            ans = input().lower()
            if ans == "y":
                print("[green]Create a detector list file.[/green]")
                break
            elif ans == "n":
                print("[green]No detector list file will be created.[/green]")
                break
            else:
                print("[ref]Invalid input. Please enter 'y' or 'n'.[/red]")
                print("[green]Do you want to make a detector list file? [y/n][/green]")
        return ans

    def extract_location_from_toml(self, file_path):
        """Extract the location of IMo from the toml file.

        Args:
            file_path (str): The path to the toml file.
        """
        with open(file_path, "r") as file:
            data = toml.load(file)
            loc = data["repositories"][0]["location"]
        return loc

    def main(self):
        if not CONFIG_FILE_PATH.exists():
            imo = Imo(flatfile_location="json file location")
            self.imo_version = "imo version"
        else:
            IMO_ROOT_PATH = self.extract_location_from_toml(CONFIG_FILE_PATH)
            imo = Imo(flatfile_location=os.path.join(IMO_ROOT_PATH, "schema.json"))
            versions = list(imo.imoobject.releases.keys())
            versions_with_idx = [f"({i + 1}). {ver}" for i, ver in enumerate(versions)]
            print(
                f"[green]Available IMO versions:[/green] [cyan]{versions_with_idx}[/cyan]"
            )
            print("[green]Input IMO version's number: [/green]")
            version_idx = input()
            self.imo_version = versions[int(version_idx) - 1]

        sim = Simulation(random_seed=None, imo=imo)

        inst_info = sim.imo.query(f"/releases/{self.imo_version}/LMHFT/instrument_info")
        channel_list = inst_info.metadata["channel_names"]
        # add index to the channel list
        channel_list_with_idx = [
            f"({i + 1}). {channel}" for i, channel in enumerate(channel_list)
        ]

        print(
            f"[green]The availavle channels are:[/green] [cyan]{channel_list_with_idx}[/cyan]"
        )
        print("[green]Input the channel's number:[/green]")
        channel_idx = input()
        self.channel = channel_list[int(channel_idx) - 1]

        print("[green]Do you want to make a detector list file? (y/n) [/green]")
        ans = self.ask_yes_or_no()
        if ans == "y":
            print(
                "[green]Input mission duration to define a scaling factor for NET (unit: yr):[/green]"
            )
            duration_yr = float(input())
            print("[green]Specify the directory to save:[/green]")
            self.base_path = input()
            if self.base_path == "":
                self.base_path = "./"
                print("[green]The file will be saved in the current directory.[/green]")
            if self.base_path.endswith("/"):
                self.base_path = self.base_path[:-1]

        for ch in channel_list:
            channel_info = FreqChannelInfo.from_imo(
                imo=imo,
                url=f"/releases/{self.imo_version}/LMHFT/{ch}/channel_info",
            )
            for detector_name in channel_info.detector_names:
                det = DetectorInfo.from_imo(
                    imo=imo,
                    url=f"/releases/{self.imo_version}/LMHFT/{ch}/{detector_name}/detector_info",
                )
                if self.channel == ch:
                    self.channel_dets_list.append(det)
                else:
                    self.total_dets_list.append(det)

        self.x_tot = np.zeros(len(self.total_dets_list))
        self.y_tot = np.zeros(len(self.total_dets_list))
        self.x_ch = np.zeros(len(self.channel_dets_list))
        self.y_ch = np.zeros(len(self.channel_dets_list))
        self.ndets_in_channel = len(self.channel_dets_list)

        # Make a figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(1, 2, 1, aspect="equal")
        self.ax1.set_title(self.channel)
        self.ax1.plot()
        self.scatter = []

        for i, det in enumerate(self.total_dets_list):
            self.x_tot[i], self.y_tot[i] = self.get_det_xy(det, self.telescope)
        self.ax1.scatter(self.x_tot, self.y_tot, marker="x", s=25, color="black")

        for i, det in enumerate(self.channel_dets_list):
            self.x_ch[i], self.y_ch[i] = self.get_det_xy(det, self.telescope)
            self.scatter.append(
                self.ax1.plot(
                    self.x_ch[i], self.y_ch[i], "o", markersize=8, color="#1f77b4"
                )[0]
            )

        self.ax1.set_xlabel(r"$\theta\cos(\phi)$ [degrees]")
        self.ax1.set_ylabel(r"$\theta\sin(\phi)$ [degrees]")

        self.ax2 = self.fig.add_subplot(1, 2, 2, aspect="equal")
        self.info_box = self.ax2.text(
            0.02,
            0.98,
            "",
            transform=self.ax2.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        self.ax2.set_axis_off()

        print("[blue]Click is available...[/blue]")
        self.fig.canvas.mpl_connect("button_press_event", self.on_plot_click)
        plt.tight_layout()
        plt.show()
        plt.savefig("testfp.png")

        # Save the detector list file.
        if ans == "y":
            filename = "detectors_LMHFT" + "_" + self.channel + "_T+B.txt"
            print(self.selected_detector_list)
            self.selected_detector_list = sorted(
                self.selected_detector_list, key=lambda detector: detector.name
            )
            self.generate_dets_list_file(
                filename, self.selected_detector_list, duration_yr
            )


if __name__ == "__main__":
    viewer = DetectorInfoViewer()
    viewer.main()
