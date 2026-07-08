# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | ------: | --------: |
| litebird\_sim/\_\_init\_\_.py                    |       42 |        0 |    100% |           |
| litebird\_sim/bandpass\_template\_module.py      |       73 |       64 |     12% |20-27, 41-45, 64-91, 104, 133-219 |
| litebird\_sim/bandpasses.py                      |      127 |       23 |     82% |101, 117, 129-130, 235-238, 246-247, 256-257, 268, 314-315, 319-332, 347 |
| litebird\_sim/beam\_convolution.py               |      113 |       13 |     88% |129-133, 180, 415-417, 422-424, 438, 455-462 |
| litebird\_sim/beam\_synthesis.py                 |      107 |        8 |     93% |255-259, 304, 311-312, 325 |
| litebird\_sim/compress.py                        |       24 |        8 |     67% | 17, 27-33 |
| litebird\_sim/constants.py                       |       16 |        0 |    100% |           |
| litebird\_sim/coordinates.py                     |       65 |       22 |     66% |110-114, 146-155, 183-193, 213-231 |
| litebird\_sim/detectors.py                       |      222 |       10 |     95% |16, 204-208, 265, 269-274, 435, 456, 587 |
| litebird\_sim/dipole.py                          |      336 |      143 |     57% |150, 153, 159-169, 305, 311, 317, 323, 329-331, 337-340, 357-381, 386-387, 392-397, 402-405, 415-416, 421-424, 433-434, 441-447, 477-483, 505-512, 539-565, 577-634, 656-677, 753, 781, 791, 889, 899, 909-918, 933-941, 952 |
| litebird\_sim/distribute.py                      |       75 |        9 |     88% |   113-122 |
| litebird\_sim/gaindrifts.py                      |      121 |       12 |     90% |239, 340, 392-396, 450, 468, 474, 549, 552, 556 |
| litebird\_sim/grasp2alm.py                       |      365 |      106 |     71% |92, 143-161, 173-197, 269-287, 303-317, 354, 364, 386-396, 405-428, 492-493, 500-501, 504, 511-512, 528, 535, 562, 566, 596, 610, 623-634, 694, 699, 712, 720, 728, 736, 803, 809-812 |
| litebird\_sim/healpix.py                         |      115 |       37 |     68% |52, 56, 99-100, 104-133, 215, 257, 271, 297 |
| litebird\_sim/hwp.py                             |       72 |       30 |     58% |35, 65, 102, 114, 119, 124, 133-144, 150-171, 208, 259 |
| litebird\_sim/hwp\_diff\_emiss.py                |       33 |        6 |     82% |15, 22-23, 49-51, 87 |
| litebird\_sim/hwp\_harmonics/\_\_init\_\_.py     |        0 |        0 |    100% |           |
| litebird\_sim/hwp\_harmonics/common.py           |       32 |       20 |     38% |8-10, 17-21, 28-32, 55-64, 94-122 |
| litebird\_sim/hwp\_harmonics/hwp\_harmonics.py   |      215 |       87 |     60% |42, 46-49, 54-63, 99, 120-124, 215-225, 233, 236-261, 292-311, 336, 348-357, 365-368, 378-382, 406-415, 437-439, 461-463, 474-488, 494, 591, 596, 652 |
| litebird\_sim/hwp\_harmonics/jones\_methods.py   |       84 |       72 |     14% |14-175, 180-201, 232-273, 306-375 |
| litebird\_sim/hwp\_harmonics/mueller\_methods.py |       15 |        8 |     47% |     47-90 |
| litebird\_sim/hwp\_jones\_parameters.py          |       32 |        7 |     78% |50, 74-77, 82-83 |
| litebird\_sim/hwp\_non\_ideal.py                 |       31 |        9 |     71% |61, 75-84, 87-90 |
| litebird\_sim/imo/\_\_init\_\_.py                |        3 |        0 |    100% |           |
| litebird\_sim/imo/imo.py                         |       81 |       21 |     74% |38-53, 58-64, 78, 96, 100, 113, 148 |
| litebird\_sim/imobrowser.py                      |      187 |      141 |     25% |29, 61-70, 75-83, 90-106, 109-112, 115-118, 121-124, 127-131, 134-138, 141-157, 160-174, 177-178, 181, 186-233, 236, 239, 242, 245-250, 253-256, 259-263, 267, 272-307, 310-313, 316-322, 325-331, 334, 338-343, 347-365 |
| litebird\_sim/input\_sky.py                      |      326 |       34 |     90% |77, 81, 87, 115-118, 123, 257, 261, 268, 292, 299, 305, 313, 320-324, 327, 340-347, 361, 367, 387, 545, 557, 617, 683, 691, 702, 747, 770, 789-790, 798 |
| litebird\_sim/install\_imo.py                    |      114 |       95 |     17% |23-32, 36-49, 59, 69-172, 181-233, 237-246, 255-273, 277 |
| litebird\_sim/io.py                              |      259 |       26 |     90% |66, 70, 227-228, 235, 239, 247, 273-274, 297-299, 478, 481, 504-505, 533, 536, 596, 626, 634, 659-661, 756-758, 806 |
| litebird\_sim/madam.py                           |      153 |       11 |     93% |295, 326-329, 349-351, 376, 398, 422, 508 |
| litebird\_sim/mapmaking/\_\_init\_\_.py          |        7 |        0 |    100% |           |
| litebird\_sim/mapmaking/binner.py                |      113 |       44 |     61% |83-91, 117-157, 168-178, 269, 435 |
| litebird\_sim/mapmaking/brahmap\_gls.py          |       13 |        0 |    100% |           |
| litebird\_sim/mapmaking/common.py                |      252 |      160 |     37% |105, 168, 182-183, 202-229, 242-261, 277-286, 322-327, 350-359, 381-447, 461-472, 476-477, 484-485, 495-499, 505-509, 533-558, 562-571, 575-586 |
| litebird\_sim/mapmaking/destriper.py             |      565 |      172 |     70% |115-166, 204, 381-408, 420-432, 533-538, 547-549, 575-599, 626-650, 663-678, 774-778, 805-832, 861-888, 1003, 1020, 1142-1148, 1338-1339, 1341-1342, 1382-1392, 1673, 1734-1740, 1752-1754, 1808-1814, 2135, 2147, 2166, 2241, 2251 |
| litebird\_sim/mapmaking/h\_maps.py               |      138 |      101 |     27% |60-63, 104-121, 142-148, 161-178, 184-188, 194-200, 213-228, 276-382, 399-414 |
| litebird\_sim/mapmaking/pair\_differencing.py    |      146 |       45 |     69% |83-91, 118-160, 211, 247-257, 280-281, 300-301, 307-308, 327-329, 354, 525 |
| litebird\_sim/maps\_and\_harmonics.py            |     1264 |      401 |     68% |128, 136, 144, 155, 163, 168, 193, 195, 217, 231, 297, 312, 324, 363-370, 392, 491, 496, 502-510, 527, 550-570, 573-577, 615, 622-627, 655, 680, 683, 691, 696, 703, 731, 734, 739, 742, 747, 756, 760, 764, 770, 773, 778, 781, 786, 792, 817-850, 879, 885, 910, 952-955, 961, 969-970, 974, 994, 998, 1012-1014, 1019-1020, 1040, 1045, 1049, 1053, 1058-1062, 1082-1087, 1127, 1143, 1165-1197, 1232-1244, 1270, 1275-1278, 1281-1284, 1287-1290, 1293-1296, 1300-1301, 1305-1308, 1312-1315, 1359-1376, 1397-1410, 1453-1454, 1482-1502, 1505, 1522-1525, 1622, 1631, 1636, 1646, 1666, 1673, 1688, 1695, 1697, 1776, 1782-1783, 1787, 1790, 1922, 1929-1934, 1986, 1994, 2002, 2040, 2043, 2048, 2051, 2056, 2065, 2069, 2073, 2080, 2083, 2088, 2091, 2096, 2103, 2129-2162, 2208, 2249, 2275, 2281-2284, 2312, 2316-2319, 2322-2325, 2328-2331, 2334-2337, 2341-2342, 2346-2349, 2353-2356, 2432, 2441, 2448, 2453, 2467, 2483, 2488-2499, 2610, 2621, 2625, 2632, 2640, 2644, 2831-2836, 2840, 2848, 2860, 2870, 2874, 2883-2886, 2896-2910, 2955-2969, 3134, 3187-3189, 3254, 3286, 3289-3295, 3381-3411, 3469, 3497-3501, 3515-3519, 3527, 3533, 3617, 3624-3627, 3636-3638, 3658-3663, 3672-3679, 3732-3734, 3758-3793, 3850-3867, 3922-3925, 3933 |
| litebird\_sim/mpi.py                             |       46 |        9 |     80% |9-13, 105-109 |
| litebird\_sim/mueller\_convolver.py              |      144 |        5 |     97% |207-208, 321, 376, 390 |
| litebird\_sim/noise.py                           |      187 |       30 |     84% |20-26, 35-41, 71-84, 273-287, 315, 443-445, 717-735, 768, 859 |
| litebird\_sim/non\_linearity.py                  |       51 |       14 |     73% |34-57, 96, 203, 206, 210 |
| litebird\_sim/observations.py                    |      379 |      148 |     61% |218-221, 232-233, 252, 259-260, 286, 299-300, 349-350, 354, 438-456, 461, 463, 470, 475, 505, 541-542, 556-698, 725-727, 762-796, 887, 1089, 1100, 1190, 1208, 1271-1297 |
| litebird\_sim/plot\_fp.py                        |      185 |      162 |     12% |22-38, 50-66, 74-97, 105-129, 143-158, 166-197, 200-211, 219-222, 225-339, 345-346 |
| litebird\_sim/pointing\_sys.py                   |      193 |       31 |     84% |39, 51, 90, 93-98, 112, 115-119, 132-134, 139-144, 483-485, 555, 562-573 |
| litebird\_sim/pointings.py                       |       64 |        4 |     94% |262, 265-268, 281 |
| litebird\_sim/pointings\_in\_obs.py              |       95 |       16 |     83% |104, 127-133, 170, 274-287, 342-347 |
| litebird\_sim/profiler.py                        |       38 |        1 |     97% |        66 |
| litebird\_sim/quaternions.py                     |       21 |        0 |    100% |           |
| litebird\_sim/scan\_map.py                       |      129 |       52 |     60% |27-29, 35-40, 45-48, 54, 61-62, 77, 92-103, 219, 222, 230, 237-238, 278-280, 289-303, 330-351, 483-489, 499, 505 |
| litebird\_sim/scanning.py                        |      189 |       30 |     84% |40, 117-123, 169-190, 210-213, 288-293, 325-326, 490, 539, 614, 703, 816, 829, 932, 977 |
| litebird\_sim/seeding.py                         |      172 |       28 |     84% |41-55, 95, 109, 140, 225, 229, 232, 239, 248, 250, 257, 263, 266, 273-274, 277, 283, 329 |
| litebird\_sim/simulations.py                     |      887 |      269 |     70% |106, 112, 124, 132, 231-259, 419, 446, 449, 487-499, 619-620, 626, 668, 689-690, 693, 698, 703, 793, 816, 838, 889-898, 908, 947, 1037-1039, 1065, 1082, 1085, 1090, 1251-1254, 1290, 1358, 1365-1372, 1402, 1446-1448, 1492-1493, 1549, 1706-1714, 1738-1759, 1828-1838, 1887, 1943, 1948, 2029, 2038, 2071-2081, 2109, 2120-2130, 2225-2230, 2259-2281, 2313, 2315, 2317, 2333-2365, 2405-2471, 2489-2507, 2537, 2555-2564, 2594-2686, 2709-2741, 2750-2783, 2913-2919, 2967 |
| litebird\_sim/spacecraft.py                      |      111 |       28 |     75% |22, 90-117, 147-203, 302, 307 |
| litebird\_sim/spherical\_harmonics.py            |        4 |        0 |    100% |           |
| litebird\_sim/units.py                           |       51 |       10 |     80% |56, 61, 73, 99, 133, 136, 158-169 |
| litebird\_sim/utilities.py                       |        6 |        0 |    100% |           |
| litebird\_sim/version.py                         |        2 |        0 |    100% |           |
| **TOTAL**                                        | **8890** | **2782** | **69%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/litebird/litebird_sim/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/litebird/litebird_sim/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Flitebird%2Flitebird_sim%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.