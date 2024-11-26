.. _quaternions-chapter:

Quaternions
===========

Rotations are important in defining a scanning strategy. Here, we
present a short tutorial on quaternions and explain the facilities
provided by LBS.

There are several choices to describe a rotation in 3D space: `Euler
angles <https://en.wikipedia.org/wiki/Euler_angles>`_, `rotation
matrices <https://en.wikipedia.org/wiki/Rotation_matrix>`_,
`quaternions <https://en.wikipedia.org/wiki/Quaternion>`_, etc. Each
of these systems has its share of advantages and disadvantages. For
instance, rotation matrices are handy when you have a vector and want
to rotate it, as it is just a matter of doing a matrix-vector
multiplication. Quaternions are more complicated in this regard, but
they offer a mathematical operation called *slerp* (shorthand for
*spherical linear interpolation*) that is not available with other
representations, like rotation matrices. We assume that the reader
knows what quaternions are and their mathematical properties; if you
are not, be sure to read the book *Visualizing quaternions*, by
Andrew J. Hanson (Elsevier, 2006, ISBN-0080474772) and the provocative
essay by Marc ten Bosch, `Let’s remove Quaternions from every 3D
engine <https://marctenbosch.com/quaternions/>`_.

The LiteBIRD simulation framework models quaternions using the
convention :math:`(v_x, v_y, v_z, w)`; be aware that some textbooks
use the order :math:`(w, v_x, v_y, v_z)`. As the framework uses
quaternions only to model rotations, they all must obey the relation
:math:`v_x^2 + v_y^2 + v_z^2 + w^2 = 1` (*normalized* quaternions),
which is a property satisfied by rotation quaternions.

The class :class:`.RotQuaternion` can model time-varying quaternions.
It is enough to provide a list of quaternions, a starting time, and a
sampling frequency, which is assumed to be constant::

    import litebird_sim as lbs

    time_varying_quaternion = lbs.RotQuaternion(
        # Three rotation quaternions
        quats=np.array(
            [
                [0.5, 0.0, 0.0, 0.8660254],
                [0.0, -0.38268343, 0.0, 0.92387953],
                [0.0, 0.0, 0.30901699, 0.95105652],
            ]
        ),
        start_time=0.0,
        sampling_rate_hz=1.0,
    )


This example assumes that ``time_varying_quaternion`` describes a
rotation that evolves with time, starting from ``t = 0`` and lasting
3 seconds, as the sampling frequency is 1 Hz.

Rotation quaternions can be multiplied together; however, they must refer
to the same starting time and have the same sampling frequency.


Python functions for quaternions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LiteBIRD Simulation Framework provides three functions,
:func:`.quat_rotation_x`, :func:`.quat_rotation_y`, :func:`.quat_rotation_z` to
compute simple rotation quaternions; they return plain the normalized
quaternion representing a rotation by an angle :math:`\theta` around
one of the three axis `x`, `y`, and `z`. These quaternions are plain
NumPy arrays and can be passed to the parameter ``quats`` of the
constructor for :class:`.RotQuaternion`:

.. testcode::

  import litebird_sim as lbs
  import numpy as np

  def print_quaternion(q):
      print("{:.3f} {:.3f} {:.3f} {:.3f}".format(*q))

  print("Rotation by π/3 around x:")
  print_quaternion(lbs.quat_rotation_x(theta_rad=np.pi/3))
  print("Rotation by π/3 around y:")
  print_quaternion(lbs.quat_rotation_y(theta_rad=np.pi/3))
  print("Rotation by π/3 around z:")
  print_quaternion(lbs.quat_rotation_z(theta_rad=np.pi/3))

.. testoutput::

   Rotation by π/3 around x:
   0.500 0.000 0.000 0.866
   Rotation by π/3 around y:
   0.000 0.500 0.000 0.866
   Rotation by π/3 around z:
   0.000 0.000 0.500 0.866


There are two functions that implement in-place multiplication of
quaternions: :func:`.quat_right_multiply` performs the calculation
:math:`r \leftarrow r \times q`, and :func:`.quat_left_multiply`
performs the calculation :math:`r \leftarrow q \times r` (where
:math:`\leftarrow` indicates the assignment operation):

.. testcode::

  quat = np.array(lbs.quat_rotation_x(np.pi / 3))
  lbs.quat_left_multiply(quat, *lbs.quat_rotation_z(np.pi / 2))
  print("Rotation by π/3 around x and then by π/2 around z:")
  print_quaternion(quat)

.. testoutput::

   Rotation by π/3 around x and then by π/2 around z:
   0.354 0.354 0.612 0.612

Note the syntax for :func:`.quat_left_multiply`: you are supposed to
pass the four components of the quaternion :math:`q` as separate
arguments, and thus we need to prepend the call to
``lbs.quat_rotation_z`` with ``*`` to expand the result (a 4-element
tuple) into the four parameters required by
:func:`.quat_left_multiply`. The reason for this weird syntax is
efficiency, as Numba (which is used extensively in the code) can
easily optimize this kind of function call.

Finally, the framework provides the function :func:`.rotate_vector`,
which applies the rotation described by a normalized quaternion to a
vector. There are faster versions in :func:`.rotate_x_vector`,
:func:`.rotate_y_vector`, and :func:`.rotate_z_vector` that rotate the
three basis vectors ``(1, 0, 0)``, ``(0, 1, 0)``, and ``(0, 0, 1)``.
The functions :func:`.all_rotate_vectors`,
:func:`.all_rotate_x_vectors`, :func:`.all_rotate_y_vectors`, and
:func:`.all_rotate_z_vectors` can be applied to arrays of quaternions
and vectors.


API reference
-------------

.. automodule:: litebird_sim.quaternions
    :members:
    :undoc-members:
    :show-inheritance:
