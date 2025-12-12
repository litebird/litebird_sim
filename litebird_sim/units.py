from enum import Enum

Units = Enum(
    "Units",
    [
        "K_CMB",
        "uK_CMB",
        "K_RJ",
        "uK_RJ",
        "MJy/sr",
        "Jy/sr",
        "ADU",
        "Pure",  # when no physical unit applies
    ],
)
