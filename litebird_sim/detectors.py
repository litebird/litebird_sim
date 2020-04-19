# -*- encoding: utf-8 -*-


class Detector:
    def __init__(
        self, imo=None, name=None, beam_z=None, sampfreq_hz=None, simulation=None
    ):
        # Basic check
        if (not imo) and (not name) and (not beam_z) and not (sampfreq_hz):
            raise ValueError(
                "You must provide some arguments to a Detector's constructor"
            )

        self.simulation = simulation

        if imo:
            assert self.simulation, (
                "You must provide a Simulation object if you plan to "
                "interface to the IMO"
            )

            metadata, obj = self.simulation.query_object(imo)

        else:
            self.name = name
            self.beam_z = beam_z
            self.sampfreq_hz = sampfreq_hz

    def read_def_from_imo(self, imo_url):
        raise NotImplementedError()
