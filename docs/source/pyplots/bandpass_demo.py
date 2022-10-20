import litebird_sim as lbs
import matplotlib.pylab as plt

# Generate the «model» (i.e., ideal band, with no wiggles)
band = lbs.BandPassInfo(
    bandcenter_ghz=43.0,
    bandwidth_ghz=10.0,
    bandtype="top-hat-cosine",
    normalize=True,
    nsamples_inband=100,
)
plt.plot(band.freqs_ghz, band.weights, label="Ideal band")

# Now generate a more realistic band with random wiggles
new_weights = band.bandpass_resampling()
plt.plot(
    band.freqs_ghz,
    new_weights,
    label="Random realization",
)

plt.xlabel("Frequency [GHz]")
plt.ylabel("Weight")
plt.legend()
