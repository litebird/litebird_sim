# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | ------: | --------: |
| litebird\_sim/\_\_init\_\_.py                    |       42 |        0 |    100% |           |
| litebird\_sim/bandpass\_template\_module.py      |       73 |       64 |     12% |20-27, 41-45, 64-91, 104, 133-219 |
| litebird\_sim/bandpasses.py                      |      127 |       23 |     82% |101, 117, 129-130, 235-238, 246-247, 256-257, 268, 314-315, 319-332, 347 |
| litebird\_sim/beam\_convolution.py               |      109 |       17 |     84% |129-133, 161-163, 180, 300, 410-412, 417-419, 433, 450-457 |
| litebird\_sim/beam\_synthesis.py                 |      107 |        8 |     93% |255-259, 304, 311-312, 325 |
| litebird\_sim/compress.py                        |       24 |        8 |     67% | 17, 27-33 |
| litebird\_sim/constants.py                       |       16 |        0 |    100% |           |
| litebird\_sim/coordinates.py                     |       65 |       22 |     66% |110-114, 146-155, 183-193, 213-231 |
| litebird\_sim/detectors.py                       |      222 |       10 |     95% |16, 204-208, 265, 269-274, 435, 456, 583 |
| litebird\_sim/dipole.py                          |      112 |       42 |     62% |61, 67-69, 75-78, 83-84, 89-94, 99-102, 112-113, 120-125, 139-185, 250, 260, 270-279 |
| litebird\_sim/distribute.py                      |       75 |        9 |     88% |   113-122 |
| litebird\_sim/gaindrifts.py                      |      121 |       12 |     90% |239, 340, 392-396, 450, 468, 474, 549, 552, 556 |
| litebird\_sim/grasp2alm.py                       |      365 |      106 |     71% |92, 143-161, 173-197, 269-287, 303-317, 354, 364, 386-396, 405-428, 492-493, 500-501, 504, 511-512, 528, 535, 562, 566, 596, 610, 623-634, 694, 699, 712, 720, 728, 736, 803, 809-812 |
| litebird\_sim/healpix.py                         |      115 |       37 |     68% |52, 56, 99-100, 104-133, 215, 257, 271, 297 |
| litebird\_sim/hwp.py                             |       72 |       30 |     58% |35, 65, 102, 114, 119, 124, 133-144, 150-171, 208, 259 |
| litebird\_sim/hwp\_diff\_emiss.py                |       33 |        6 |     82% |15, 22-23, 49-51, 87 |
| litebird\_sim/hwp\_harmonics/\_\_init\_\_.py     |        0 |        0 |    100% |           |
| litebird\_sim/hwp\_harmonics/common.py           |       32 |       20 |     38% |8-10, 17-21, 28-32, 55-64, 94-122 |
| litebird\_sim/hwp\_harmonics/hwp\_harmonics.py   |      212 |       81 |     62% |38, 50-59, 95, 111-114, 216-226, 234, 237-262, 293-312, 337, 349-358, 366-369, 379-383, 407-416, 438-440, 462-464, 475-489, 495, 593, 649 |
| litebird\_sim/hwp\_harmonics/jones\_methods.py   |       86 |       74 |     14% |14-175, 180-201, 232-275, 308-378 |
| litebird\_sim/hwp\_harmonics/mueller\_methods.py |       15 |        8 |     47% |     47-90 |
| litebird\_sim/hwp\_jones\_parameters.py          |       32 |        7 |     78% |50, 74-77, 82-83 |
| litebird\_sim/hwp\_non\_ideal.py                 |       31 |        9 |     71% |61, 75-84, 87-90 |
| litebird\_sim/imo/\_\_init\_\_.py                |        3 |        0 |    100% |           |
| litebird\_sim/imo/imo.py                         |       81 |       21 |     74% |38-53, 58-64, 78, 96, 100, 113, 148 |
| litebird\_sim/imobrowser.py                      |      187 |      141 |     25% |29, 61-70, 75-83, 90-106, 109-112, 115-118, 121-124, 127-131, 134-138, 141-157, 160-174, 177-178, 181, 186-233, 236, 239, 242, 245-250, 253-256, 259-263, 267, 272-307, 310-313, 316-322, 325-331, 334, 338-343, 347-365 |
| litebird\_sim/input\_sky.py                      |      325 |       34 |     90% |77, 81, 87, 115-118, 123, 248, 252, 259, 283, 290, 296, 304, 311-315, 318, 331-338, 352, 358, 378, 536, 548, 608, 674, 682, 693, 745, 768, 787-788, 796 |
| litebird\_sim/install\_imo.py                    |      114 |       95 |     17% |23-32, 36-49, 59, 69-172, 181-233, 237-246, 255-273, 277 |
| litebird\_sim/io.py                              |      259 |       26 |     90% |66, 70, 227-228, 235, 239, 247, 273-274, 297-299, 478, 481, 504-505, 533, 536, 596, 626, 634, 659-661, 756-758, 806 |
| litebird\_sim/madam.py                           |      153 |       11 |     93% |295, 326-329, 349-351, 376, 398, 422, 508 |
| litebird\_sim/mapmaking/\_\_init\_\_.py          |        6 |        0 |    100% |           |
| litebird\_sim/mapmaking/binner.py                |      113 |       44 |     61% |83-91, 117-157, 168-178, 269, 435 |
| litebird\_sim/mapmaking/brahmap\_gls.py          |       13 |        0 |    100% |           |
| litebird\_sim/mapmaking/common.py                |      236 |      146 |     38% |105, 176-177, 192-211, 227-236, 272-277, 300-309, 331-397, 411-422, 426-427, 434-435, 445-449, 455-459, 483-508, 512-521, 525-536 |
| litebird\_sim/mapmaking/destriper.py             |      565 |      172 |     70% |115-166, 204, 381-408, 420-432, 533-538, 547-549, 575-599, 626-650, 663-678, 774-778, 805-832, 861-888, 1003, 1020, 1142-1148, 1338-1339, 1341-1342, 1382-1392, 1673, 1734-1740, 1752-1754, 1808-1814, 2135, 2147, 2166, 2241, 2251 |
| litebird\_sim/mapmaking/pair\_differencing.py    |      146 |       45 |     69% |83-91, 118-160, 211, 247-257, 280-281, 300-301, 307-308, 327-329, 354, 525 |
| litebird\_sim/maps\_and\_harmonics.py            |     1257 |      399 |     68% |128, 136, 144, 155, 163, 168, 193, 195, 217, 231, 297, 312, 324, 363-370, 392, 469-477, 494, 517-537, 540-544, 582, 589-594, 622, 647, 650, 658, 663, 670, 698, 701, 706, 709, 714, 723, 727, 731, 737, 740, 745, 748, 753, 759, 784-817, 846, 852, 877, 919-922, 928, 936-937, 941, 961, 965, 979-981, 986-987, 1007, 1012, 1016, 1020, 1025-1029, 1049-1054, 1094, 1110, 1132-1164, 1199-1211, 1237, 1242-1245, 1248-1251, 1254-1257, 1260-1263, 1267-1268, 1272-1275, 1279-1282, 1326-1343, 1364-1377, 1420-1421, 1449-1469, 1472, 1489-1492, 1589, 1598, 1603, 1613, 1633, 1640, 1655, 1662, 1664, 1743, 1749-1750, 1754, 1757, 1889, 1896-1901, 1953, 1961, 1969, 2007, 2010, 2015, 2018, 2023, 2032, 2036, 2040, 2047, 2050, 2055, 2058, 2063, 2070, 2096-2129, 2175, 2216, 2242, 2248-2251, 2279, 2283-2286, 2289-2292, 2295-2298, 2301-2304, 2308-2309, 2313-2316, 2320-2323, 2399, 2408, 2415, 2420, 2434, 2450, 2455-2466, 2577, 2588, 2592, 2599, 2607, 2611, 2798-2803, 2807, 2815, 2827, 2837, 2841, 2850-2853, 2863-2877, 2922-2936, 3101, 3154-3156, 3221, 3253, 3256-3262, 3348-3378, 3436, 3464-3468, 3482-3486, 3494, 3500, 3584, 3591-3594, 3603-3605, 3625-3630, 3639-3646, 3699-3701, 3725-3760, 3817-3834, 3889-3892, 3900 |
| litebird\_sim/mpi.py                             |       46 |        9 |     80% |9-13, 105-109 |
| litebird\_sim/mueller\_convolver.py              |      144 |        5 |     97% |207-208, 321, 376, 390 |
| litebird\_sim/noise.py                           |       79 |       22 |     72% |20-26, 35-41, 71-84, 214-228, 395, 398 |
| litebird\_sim/non\_linearity.py                  |       43 |       14 |     67% |32-55, 89, 176, 179, 183 |
| litebird\_sim/observations.py                    |      380 |      148 |     61% |225-228, 239-240, 259, 266-267, 293, 306-307, 356-357, 361, 445-463, 468, 470, 477, 482, 512, 548-549, 563-705, 732-734, 769-803, 894, 1096, 1107, 1197, 1215, 1278-1304 |
| litebird\_sim/plot\_fp.py                        |      185 |      162 |     12% |22-38, 50-66, 74-97, 105-129, 143-158, 166-197, 200-211, 219-222, 225-339, 345-346 |
| litebird\_sim/pointing\_sys.py                   |      193 |       31 |     84% |39, 51, 90, 93-98, 112, 115-119, 132-134, 139-144, 483-485, 555, 562-573 |
| litebird\_sim/pointings.py                       |       64 |        4 |     94% |262, 265-268, 281 |
| litebird\_sim/pointings\_in\_obs.py              |       95 |       16 |     83% |104, 127-133, 170, 274-287, 342-347 |
| litebird\_sim/profiler.py                        |       38 |        1 |     97% |        66 |
| litebird\_sim/quaternions.py                     |       21 |        0 |    100% |           |
| litebird\_sim/scan\_map.py                       |      129 |       52 |     60% |27-29, 35-40, 45-48, 54, 61-62, 77, 92-103, 224, 227, 235, 242-243, 283-285, 294-308, 335-356, 494-500, 510, 516 |
| litebird\_sim/scanning.py                        |      189 |       30 |     84% |40, 117-123, 169-190, 210-213, 288-293, 325-326, 490, 539, 614, 703, 816, 829, 932, 977 |
| litebird\_sim/seeding.py                         |      175 |       28 |     84% |39-53, 93, 107, 138, 225, 229, 232, 239, 248, 250, 257, 263, 266, 273-274, 277, 283, 329 |
| litebird\_sim/simulations.py                     |      886 |      268 |     70% |104, 110, 122, 130, 229-257, 417, 444, 447, 485-497, 617-618, 624, 666, 687-688, 691, 696, 701, 791, 814, 836, 887-896, 906, 945, 1035-1037, 1063, 1080, 1083, 1088, 1249-1252, 1288, 1356, 1363-1370, 1400, 1444-1446, 1490-1491, 1547, 1704-1712, 1736-1757, 1832-1842, 1891, 1947, 1952, 2036, 2045, 2078-2088, 2115, 2126-2136, 2191-2196, 2225-2247, 2279, 2281, 2283, 2299-2331, 2371-2437, 2455-2473, 2488-2497, 2527-2619, 2642-2674, 2683-2716, 2845-2851, 2899 |
| litebird\_sim/spacecraft.py                      |      111 |       28 |     75% |22, 90-117, 147-203, 302, 307 |
| litebird\_sim/spherical\_harmonics.py            |        4 |        0 |    100% |           |
| litebird\_sim/units.py                           |       49 |       19 |     61% |54, 59, 71, 97, 130-163 |
| litebird\_sim/utilities.py                       |        6 |        0 |    100% |           |
| litebird\_sim/version.py                         |        2 |        0 |    100% |           |
| **TOTAL**                                        | **8383** | **2564** | **69%** |           |


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