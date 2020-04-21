# -*- encoding: utf-8 -*-

import os


class MpiWrapper:
    """Wrap optional MPI functionalities into a class.

    This class is used to create the global variable
    :data:`MPI_COMM_WORLD`, which should be used whenever the code
    provides optional support for MPI.

    The class behaves like ``mpi4py.MPI``, if ``mpi4py`` can be
    imported, wrapping all the usual functions (``sendv``, ``recv``,
    ``gather``, ``reduce``, etc.). It provides a few additional
    members:

    - ``configuration`` (dict): the result of ``mpi4py.get_config``
    - ``have_mpi`` (bool): ``True`` if ``mpi4py`` is being used,
    - ``False`` otherwise.

    If no MPI is available, the class defines the member variables
    ``rank`` to 0 and ``size`` to 1. Using methods from
    ``mpi4py.MPI.COMM_WORLD`` will result in a error.

    The behavior of this class is affected by the environment
    variables ``LITEBIRD_SIM_MPI`` and ``ENABLE_MPI``, which are
    checked in this order: if they are set to a null value or a true
    Boolean string (either ``1``, ``yes``, ``true``, or ``on``,
    case-insensitive), then the presence of ``mpi4py`` is forced and
    an exception is thrown if ``mpi4py`` cannot be imported. If they
    are set to a false Boolean string, then ``mpi4py`` is not
    imported. If no environment variable is found, the code tries to
    import ``mpi4py`` but does not complain if it fails to do so.

    """

    def __init__(self):
        self.have_mpi = False
        self.rank = 0
        self.size = 1
        self.configuration = {}
        self.comm_world = None

        env_enable = self._check_if_enable_mpi()
        if env_enable in (None, True):
            try:
                import mpi4py
                from mpi4py import MPI

                self.have_mpi = True
                self.rank = MPI.COMM_WORLD.rank
                self.size = MPI.COMM_WORLD.size
                self.configuration = mpi4py.get_config()
                self.comm_world = MPI.COMM_WORLD
            except ModuleNotFoundError:
                if env_enable:
                    raise  # Rethrow exception

    @staticmethod
    def _check_if_enable_mpi():
        for keyword in ["LITEBIRD_SIM_MPI", "ENABLE_MPI"]:
            if keyword not in os.environ:
                continue

            value = os.environ[keyword]
            if value.lower() in ["1", "yes", "true", "on", ""]:
                return True
            if value.lower() in ["0", "no", "false", "off"]:
                return False

        return None

    def __getattr__(self, attr):
        # Forward any call to self.comm_world, which is equal to
        # MPI.COMM_WORLD; in this way, MpiWrapper can respond to any
        # method call that is available by MPI.COMM_WORLD (e.g.,
        # `barrier`, `allreduce`, etc.)
        if attr in dir(self.comm_world):
            return getattr(self.comm_world, attr)
        else:
            raise AttributeError(
                (
                    "Neither 'MpiWrapper' nor 'MPI.COMM_WORLD' "
                    + "have a member named '{}'"
                ).format(attr)
            )


#: Global variable of type :class:`MpiWrapper`. Use this whenever you
#: might want to use MPI
MPI_COMM_WORLD = MpiWrapper()
