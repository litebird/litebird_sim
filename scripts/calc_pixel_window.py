import healpy as hp
import pickle


def compute_pixwin_dict(min_nside_pow=1, max_nside_pow=13):
    """
    Computes HEALPix pixel window functions for a range of Nside values.

    Args:
        min_nside_pow (int): Power of 2 for starting Nside (default 1 -> 2^1 = 2)
        max_nside_pow (int): Power of 2 for ending Nside (default 13 -> 2^13 = 8192)

    Returns:
        dict: Dictionary where keys are Nside (int) and values are
              dictionaries containing 'T' and 'E' arrays.
    """

    # Generate list of Nsides: [2, 4, 8, ... , 8192]
    nside_list = [2**i for i in range(min_nside_pow, max_nside_pow + 1)]

    pw_library = {}

    print(
        f"Computing Pixel Window Functions for Nside {nside_list[0]} to {nside_list[-1]}..."
    )
    print("-" * 60)

    for nside in nside_list:
        # Define lmax. Standard convention is usually 3*nside or 4*nside.
        # healpy.pixwin defaults ensuring coverage for the pixel size.
        # We explicitly request Polarization (pol=True).

        # hp.pixwin returns a tuple of arrays: (Window_T, Window_Pol)
        # Note: In HEALPix, the window function for E and B is identical.
        pw_t, pw_pol = hp.pixwin(nside, pol=True, lmax=4 * nside)

        pw_library[nside] = {
            "T": pw_t,
            "E": pw_pol,  # Assigning Pol window to E as requested
            "lmax": len(pw_t) - 1,
        }

        print(f"Nside: {nside:<5} | lmax: {len(pw_t) - 1:<5} | Computed T & E")

    return pw_library


if __name__ == "__main__":
    # 1. Compute the dictionary
    window_function_dict = compute_pixwin_dict()

    # 2. Save to disk (Optional but recommended for large simulations)
    output_filename = "litebird_pixwin_db.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(window_function_dict, f)

    print("-" * 60)
    print(f"✅ Dictionary saved to {output_filename}")

    # 3. Usage Example Verification
    sample_nside = 128
    print(f"\nVerifying content for Nside={sample_nside}:")
    print(f"T Window shape: {window_function_dict[sample_nside]['T'].shape}")
    print(f"E Window shape: {window_function_dict[sample_nside]['E'].shape}")
    print(f"Value at ell=10 (T): {window_function_dict[sample_nside]['T'][10]:.5f}")
