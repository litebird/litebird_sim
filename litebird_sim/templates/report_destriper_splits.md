## Destriper for detector split = "{{ detector_split }}" and time split = "{{ time_split }}"

### Input parameters

The destriper was executed with the following parameters:

- `NSIDE = {{ results.nside }}`
- Coordinate system for the output map: `{{ results.params.output_coordinate_system }}`
- Samples per baseline: `{{ results.params.samples_per_baseline }}`
- Maximum number of iterations: `{{ results.params.iter_max }}`
- Threshold to stop the CG iteration: `{{ results.params.threshold }}`
- CG preconditioner: {% if results.params.use_preconditioner %}**on**{% else %}off{% endif %}
{% if not results.baselines %}- No destriping (only binning){% endif %}

{% if results.baselines %}
### Convergence

- The convergence was {% if results.converged %}achieved{% else %}**not** achieved{% endif %}
- Number of CG steps: `{{ results.history_of_stopping_factors | length }}`
- Memory used by temporary buffers: {{ results.bytes_in_temporary_buffers | filesizeformat }}
- Memory used by Cholesky matrices: {{ bytes_in_cholesky_matrices | filesizeformat }}
- Elapsed time: {{ "%.1f" | format(results.elapsed_time_s) }} s

<div style="text-align: center">
<img src="{{ cg_plot_filename }}">
</div>

| **Step**                                           | **Residual** [K] |
|----------------------------------------------------|------------------|
{% for cur_value in history_of_stopping_factors %}| {{loop.index}}   | {{ "%.2e" | format(cur_value) }} |
{% endfor %} 
{% endif %}
