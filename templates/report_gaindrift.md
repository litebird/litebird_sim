## Gain drift injection on TOD `{{ component }}`

{{drift_type}} gain drift has been applied on the TOD `{{ component }}` with the following parameters:

- `sigma_drift` = {{ sigma_drift }}
- `sampling_dist` =  {{ sampling_dist }}
- `user_seed` = {{ user_seed }}
- `sampling_uniform_low` = {{ sampling_uniform_low }}
- `sampling_uniform_high` = {{ sampling_uniform_high }}
- `sampling_gaussian_loc` = {{ sampling_gaussian_loc }}
- `sampling_gaussian_scale` = {{ sampling_gaussian_scale }}
{% if linear_drift -%}
- `drift_type` = {{ drift_type }}
- `linear_drift` = {{ linear_drift }}
- `calibration_period_sec` = {{ calibration_period_sec }}
{% endif %}
{% if thermal_drift -%}
- `focalplane_group` = {{ focalplane_group }}
- `oversample` = {{ oversample }}
- `fknee_drift_mHz` = {{ fknee_drift_mHz }}
- `alpha_drift` = {{ alpha_drift }}
- `detector_mismatch` = {{ detector_mismatch }}
- `thermal_fluctuation_amplitude_K` = {{ thermal_fluctuation_amplitude_K }}
- `focalplane_Tbath_K` = {{ focalplane_Tbath_K }}
{% endif %}
