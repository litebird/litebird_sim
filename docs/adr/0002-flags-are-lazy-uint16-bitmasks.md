# Flags are lazily allocated uint16 bitmasks

The flags of the samples live in `Observation.local_flags` (per detector,
`(n_detectors, n_samples)`) and `Observation.global_flags` (common to all the
detectors, `(n_samples,)`). They are bitmasks of dtype `FLAG_DTYPE = uint16`:
zero is a good sample, and each bit is a different reason why the sample is
flagged (`FLAG_PLANET`, …), defined in `litebird_sim/flags.py`. Both arrays are
`None` until somebody writes them, meaning that all the samples are good.

## Considered Options

- **Bitmask (chosen)** vs a single Boolean flag. With one bit per reason the
  same TOD gives maps with and without a class of samples (e.g. to test the
  radius of the planet flagging), the flagged samples can be kept for other
  uses (planet data for beam reconstruction) and the loss of data can be
  counted per reason. This is what Planck, TOAST and so3g do.
- **Lazy allocation (chosen)** vs allocating the flags in `Observation.__init__`.
  `uint16` flags take half the memory of a `float32` TOD, which would be paid
  by every simulation, even those that flag nothing. Writers call
  `get_local_flags`/`get_global_flags`, which allocate the arrays on the first
  write.
- **uint16** vs uint8 or uint32: 16 reasons are plenty, and uint8 would save
  memory only if we never needed more than 8.

## Consequences

Every consumer of the flags (map-makers, I/O, MPI redistribution) must accept
`None`. Modules set their own bit with `flags |= FLAG_X` and never clear the
bits of others; consumers select the reasons to exclude with
`(flags & mask) != 0`. New reasons get a new bit in `flags.py`.
