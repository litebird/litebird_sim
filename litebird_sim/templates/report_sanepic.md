## pysanepic GLS

- pysanepic version: `{{ sanepic_version }}`
- `NSIDE = {{ result.nside }}`, coordinate system: `{{ result.coordinates }}`
- Chunk length for the noise filter: {{ params.chunk_s }} s, padding {{ params.pad_s }} s (`{{ params.pad_fill }}`)
- Preconditioner: `{{ params.preconditioner }}`, `min_pol_rcond = {{ params.min_pol_rcond }}`
- PCG threshold: `{{ params.tol }}`; convergence {% if result.converged %}achieved{% else %}**not** achieved{% endif %} in {{ result.iterations }} iterations
