.. _plot_fp:

Focal plane visualization
===========

We can visualize detectors in the focal plane by:
```
python -m litebird_sim.plot_fp
```
This software loads the IMo which is installed in the machine you are using.
As the conversation unfolds, an interactive Matplotlib window will appear.
Detectors corresponding to the specified channels are represented as blue dots.
Clicking on a dot reveals the `DetectorInfo` for that detector in real time, highlighted with a red star.
Additionally, if you agree during the conversation to generate a detector file,
a list of starred detectors will be saved into a text file at the designated location after you closed the plot.

The format of the detector file is as follows:
```
# Telescope     Channel         IMO_NET         Number_det      Scaled_NET      Detector_name
LFT		L1-040		114.63		2/48		13.51		000_003_003_UB_040_B
LFT		L1-040		114.63		2/48		13.51		000_003_003_UB_040_T
```
The description of each column is as follows:
- `Telescope`: The telescope name.
- `Channel`: The channel name.
- `IMO_NET`: The NET of the detector in IMo.
- `Number_det`: :math:`N_{\text{selected}}/N_{\text{total}}` where :math:`N_{\text{selected}}` is number of selected detectors by clicking and :math:`N_{\text{total}}` is total number of detectors in the channel.
scaling_factor = np.sqrt(duration_yr/3. * selected_number_of_dets/self.ndets_in_channel)
- `Scaled_NET`: The scaled NET of the detectors by the following equation:
  .. math::
    \text{Scaled NET} = \text{NET_{IMO}} \sqrt{\frac{t\_obs}{3} \frac{N_{\text{selected}}}{N_{\text{total}}}}
where :math:`t\_obs` is the observation time in years that you can specify in the conversation.
- `Detector_name`: The detector name.
