## pysanepic GLS

- pysanepic version: `{{ sanepic_version }}`
- `NSIDE = {{ result.nside }}`, coordinate system: `{{ result.coordinates }}`
- Chunk length for the noise filter: {{ chunk_s }} s
- PCG threshold: `{{ tol }}`; convergence {% if result.converged %}achieved{% else %}**not** achieved{% endif %} in {{ result.iterations }} iterations
