# MPI

- Mpi4py version: {{ mpi4py_version }}
- MPI implementation: {{ mpi_implementation }}
- MPI interface version: {{ mpi_version }}

{% if warning_mpi_version -%}
**Warning**: you should use a MPI library compliant with MPI-4, but
your MPI implementation only supports MPI-{{ mpi_version }}.

You might encounter problems if you generate maps with NSIDE=2048, as
explained in issue [#406](https://github.com/litebird/litebird_sim/issues/406):
MPI functions started supporting blocks larger than 2GB only with MPI-4.
LBS implements a few tricks that should work even in this case,
but there might be undiscovered issues in the codebase.

If you are experiencing problems with maps with `NSIDE = 2048`, the
most likely culprit is the MPI implementation you are using, not Mpi4py.
{% endif %}


