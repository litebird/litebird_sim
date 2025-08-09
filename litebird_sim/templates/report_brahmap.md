## BrahMap GLS

- BrahMap version: `{{ brahmap_version }}`
- BrahMap Git hash: `{{ brahmap_hash }}`

### Input parameters

The BrahMap GLS was called with the following parameters:

- `NSIDE = {{ gls_result.nside }}`
- Coordinate system for the output map: `{{ gls_params.output_coordinate_system }}`
- Solver type: `{{ gls_params.solver_type }}`
- Use iterative solver: `{{ gls_params.use_iterative_solver }}`
{% if gls_params.use_iterative_solver %}
  - Maximum number of iterations: `{{ gls_params.isolver_max_iterations }}`
  - Stopping threshold for iterative solver: `{{ gls_params.isolver_threshold }}`
{% endif %}

### Convergence

- The convergence was {% if gls_result.convergence_status %}achieved{% else %}**not** achieved{% endif %}
- Number of iterations performed by the iterative solver: `{{ gls_result.num_iterations }}`
