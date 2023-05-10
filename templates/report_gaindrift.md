## Applying gaindrift

{{drift_type}} gain drift has been applied on the TODs with the following parameters:

- {{ sigma_drift_K }}
- {{ sampling_dist }}

{% if linear_drift -%}

- {{ drift_type }}
- {{ linear_drift }}
- {{ calibration_period }}

{% endif %}

{% if thermal_drift -%}

- {{ focalplane_group }}
- {{ oversample }}
- {{ fknee_drift_mHz }}
- {{ alpha_drift }}
- {{ sampling_freq_Hz }}
- {{ detector_mismatch }}
- {{ thermal_fluctuation_amplitude_K }}
- {{ focalplane_Tbath_mK }}

{% endif %}
