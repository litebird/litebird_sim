# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | ------: | --------: |
| litebird\_sim/\_\_init\_\_.py                    |       43 |        0 |    100% |           |
| litebird\_sim/bandpass\_template\_module.py      |       73 |       40 |     45% |20-27, 41-45, 64-91, 104, 147-155, 171-172, 177-178, 201, 211-218, 226 |
| litebird\_sim/bandpasses.py                      |      125 |       23 |     82% |106, 122, 134-135, 237-240, 248-249, 258-259, 270, 316-317, 321-334, 349 |
| litebird\_sim/beam\_convolution.py               |      115 |       11 |     90% |132-136, 183, 423-425, 430-432, 446, 463 |
| litebird\_sim/beam\_synthesis.py                 |      107 |        8 |     93% |255-259, 304, 311-312, 325 |
| litebird\_sim/compress.py                        |       24 |        8 |     67% | 17, 27-33 |
| litebird\_sim/constants.py                       |       16 |        0 |    100% |           |
| litebird\_sim/coordinates.py                     |       67 |       22 |     67% |116-120, 152-161, 189-199, 219-237 |
| litebird\_sim/detectors.py                       |      222 |       10 |     95% |16, 204-208, 265, 269-274, 435, 456, 587 |
| litebird\_sim/dipole.py                          |      317 |      137 |     57% |151, 154, 160-170, 306, 312, 318, 324, 330-332, 338-341, 360-384, 389-390, 395-400, 405-408, 418-419, 424-427, 436-437, 444-450, 480-486, 508-515, 542-568, 580-637, 659-680, 759, 787, 797, 911-919, 930 |
| litebird\_sim/distribute.py                      |       75 |        9 |     88% |   113-122 |
| litebird\_sim/gaindrifts.py                      |      114 |       10 |     91% |239, 340, 392-396, 450, 468, 474, 549 |
| litebird\_sim/grasp2alm.py                       |      366 |      106 |     71% |97, 148-166, 178-202, 274-292, 308-322, 359, 369, 391-401, 410-433, 497-498, 505-506, 509, 516-517, 533, 540, 567, 571, 601, 615, 628-639, 699, 704, 717, 725, 733, 741, 808, 814-817 |
| litebird\_sim/healpix.py                         |      115 |       37 |     68% |52, 56, 99-100, 104-133, 215, 257, 271, 297 |
| litebird\_sim/hwp.py                             |       72 |       30 |     58% |35, 65, 102, 114, 119, 124, 133-144, 150-171, 208, 259 |
| litebird\_sim/hwp\_diff\_emiss.py                |       32 |        5 |     84% |17, 24-25, 51-53 |
| litebird\_sim/hwp\_harmonics/\_\_init\_\_.py     |        0 |        0 |    100% |           |
| litebird\_sim/hwp\_harmonics/common.py           |       19 |       10 |     47% |7-9, 16-20, 27-31, 54-63 |
| litebird\_sim/hwp\_harmonics/hwp\_harmonics.py   |      213 |       86 |     60% |46, 50-53, 58-67, 103, 124-128, 219-229, 238-263, 294-313, 338, 350-359, 367-370, 380-384, 409-418, 440-442, 464-466, 477-492, 498, 595, 600, 656 |
| litebird\_sim/hwp\_harmonics/jones\_methods.py   |       67 |       52 |     22% |14-46, 52-61, 68-75, 106-135, 169-211 |
| litebird\_sim/hwp\_harmonics/mueller\_methods.py |       22 |       12 |     45% | 23, 61-99 |
| litebird\_sim/hwp\_jones\_parameters.py          |       32 |        7 |     78% |50, 74-77, 82-83 |
| litebird\_sim/hwp\_non\_ideal.py                 |       31 |        9 |     71% |61, 75-84, 87-90 |
| litebird\_sim/imo/\_\_init\_\_.py                |        3 |        0 |    100% |           |
| litebird\_sim/imo/imo.py                         |       81 |       21 |     74% |38-53, 58-64, 78, 96, 100, 113, 148 |
| litebird\_sim/imobrowser.py                      |      187 |      140 |     25% |61-70, 75-83, 90-106, 109-112, 115-118, 121-124, 127-131, 134-138, 141-157, 160-174, 177-178, 181, 186-233, 236, 239, 242, 245-250, 253-256, 259-263, 267, 272-307, 310-313, 316-322, 325-331, 334, 338-343, 347-365 |
| litebird\_sim/input\_sky.py                      |      324 |       32 |     90% |78, 82, 88, 111, 245, 249, 256, 280, 287, 293, 301, 308-312, 315, 328-335, 349, 355, 375, 533, 545, 605, 671, 679, 690, 735, 758, 777-778, 786 |
| litebird\_sim/install\_imo.py                    |      114 |       95 |     17% |23-32, 36-49, 59, 69-172, 181-233, 237-246, 255-273, 277 |
| litebird\_sim/io.py                              |      267 |       24 |     91% |66, 70, 243-244, 251, 255, 263, 293-294, 324-326, 505, 508, 531-532, 563, 623, 653, 661, 691-693, 791-793 |
| litebird\_sim/madam.py                           |      153 |       11 |     93% |295, 326-329, 349-351, 376, 398, 422, 508 |
| litebird\_sim/mapmaking/\_\_init\_\_.py          |        8 |        0 |    100% |           |
| litebird\_sim/mapmaking/binner.py                |      126 |       59 |     53% |88-119, 148-184, 195-205, 452 |
| litebird\_sim/mapmaking/brahmap\_gls.py          |       18 |        1 |     94% |       132 |
| litebird\_sim/mapmaking/common.py                |      273 |      152 |     44% |105, 184, 198-199, 219-249, 262-281, 297-306, 342-347, 370-379, 393-394, 416-482, 514-515, 522-523, 533-537, 543-547, 571-596, 600-609, 613-624, 653 |
| litebird\_sim/mapmaking/destriper.py             |      567 |      158 |     72% |115-166, 204, 382-411, 423-430, 530-535, 548-551, 578-603, 631-656, 669-684, 778-783, 812-840, 871-899, 1155-1161, 1360-1361, 1363-1364, 1691, 1762-1764, 1818-1824, 2145, 2157, 2176, 2251, 2261 |
| litebird\_sim/mapmaking/h\_maps.py               |      141 |      102 |     28% |62-65, 106-123, 144-150, 163-179, 185-189, 195-201, 214-229, 281-389, 406-421 |
| litebird\_sim/mapmaking/pair\_differencing.py    |      141 |       42 |     70% |84-92, 122-160, 212, 247-250, 271-272, 291-292, 298-299, 318-320, 345, 516 |
| litebird\_sim/mapmaking/sanepic\_gls.py          |      189 |      167 |     12% |44-47, 53-59, 66-72, 76-88, 121-198, 218-266, 416-497, 585-654, 696-730 |
| litebird\_sim/maps\_and\_harmonics.py            |     1274 |      401 |     69% |129, 137, 145, 156, 164, 169, 194, 196, 218, 232, 298, 313, 325, 364-371, 393, 492, 497, 503-511, 528, 551-571, 574-578, 616, 623-628, 656, 681, 684, 692, 697, 704, 732, 735, 740, 743, 748, 757, 761, 765, 771, 774, 779, 782, 787, 793, 818-851, 880, 886, 911, 953-956, 962, 970-971, 975, 995, 999, 1013-1015, 1020-1021, 1041, 1046, 1050, 1054, 1059-1063, 1083-1088, 1128, 1144, 1166-1198, 1233-1245, 1271, 1276-1279, 1282-1285, 1288-1291, 1294-1297, 1301-1302, 1306-1309, 1313-1316, 1360-1377, 1398-1411, 1454-1455, 1483-1503, 1506, 1523-1526, 1623, 1632, 1637, 1647, 1667, 1674, 1689, 1696, 1698, 1777, 1783-1784, 1788, 1791, 1923, 1930-1935, 1987, 1995, 2003, 2041, 2044, 2049, 2052, 2057, 2066, 2070, 2074, 2081, 2084, 2089, 2092, 2097, 2104, 2130-2163, 2209, 2250, 2276, 2282-2285, 2313, 2317-2320, 2323-2326, 2329-2332, 2335-2338, 2342-2343, 2347-2350, 2354-2357, 2454, 2467, 2474, 2479, 2493, 2509, 2514-2525, 2638, 2649, 2653, 2660, 2668, 2672, 2861-2866, 2870, 2878, 2890, 2900, 2904, 2913-2916, 2926-2940, 2985-2999, 3167, 3220-3222, 3287, 3319, 3322-3328, 3414-3444, 3502, 3530-3534, 3548-3552, 3560, 3566, 3650, 3657-3660, 3669-3671, 3691-3696, 3705-3712, 3765-3767, 3791-3826, 3883-3900, 3955-3958, 3966 |
| litebird\_sim/mpi.py                             |       39 |       11 |     72% |10-14, 61, 76-79, 83 |
| litebird\_sim/mueller\_convolver.py              |      144 |        5 |     97% |207-208, 321, 376, 390 |
| litebird\_sim/noise.py                           |      182 |       29 |     84% |19-25, 34-40, 70-83, 272-286, 314, 442-444, 716-734, 767 |
| litebird\_sim/non\_linearity.py                  |       45 |       12 |     73% |34-57, 96, 203 |
| litebird\_sim/observation\_utilities.py          |      123 |       13 |     89% |108, 131-137, 174, 284-297 |
| litebird\_sim/observations.py                    |      386 |      127 |     67% |247-250, 261-262, 281, 288-289, 317, 388-389, 393, 477-495, 500, 502, 544, 580-581, 595-737, 764-766, 801-826, 917, 1151, 1162, 1246, 1264, 1310, 1388 |
| litebird\_sim/plot\_fp.py                        |      185 |      162 |     12% |22-38, 50-66, 74-97, 105-129, 143-158, 166-197, 200-211, 219-222, 225-339, 345-346 |
| litebird\_sim/pointing\_sys.py                   |      193 |       31 |     84% |39, 51, 90, 93-98, 112, 115-119, 132-134, 139-144, 483-485, 555, 562-573 |
| litebird\_sim/pointings.py                       |       91 |        6 |     93% |56-57, 351, 354-357, 370 |
| litebird\_sim/profiler.py                        |       38 |        1 |     97% |        66 |
| litebird\_sim/quaternions.py                     |       21 |        0 |    100% |           |
| litebird\_sim/scan\_map.py                       |      130 |       48 |     63% |32-34, 40-45, 50-52, 58, 65-67, 82, 97-108, 226, 229, 237, 244-245, 286-288, 306-308, 312, 339-360, 492-498, 508, 514 |
| litebird\_sim/scanning.py                        |      211 |       41 |     81% |44, 121-127, 173-201, 221-224, 299-304, 336-337, 525, 574, 649, 785, 823-833, 898, 911, 1013, 1058 |
| litebird\_sim/seeding.py                         |      172 |       28 |     84% |41-55, 95, 109, 140, 225, 229, 232, 239, 248, 250, 257, 263, 266, 273-274, 277, 283, 329 |
| litebird\_sim/shared\_memory.py                  |       67 |        4 |     94% |37, 66, 81, 100 |
| litebird\_sim/simulations.py                     |      997 |      267 |     73% |119, 125, 137, 145, 260-288, 485, 523-535, 655-656, 662, 704, 725-726, 729, 734, 739, 829, 852, 874, 925-934, 944, 983, 1073-1075, 1101, 1118, 1121, 1126, 1287-1290, 1326, 1379-1380, 1427, 1437, 1452, 1495, 1498, 1519, 1526-1533, 1563, 1653-1654, 1718, 1766, 1812, 1917, 1969-1977, 2001-2022, 2091-2101, 2150, 2206, 2211, 2292, 2301, 2334-2344, 2372, 2383-2393, 2488-2493, 2522-2544, 2576, 2578, 2580, 2596-2628, 2668-2734, 2755-2756, 2758, 2800, 2818-2827, 2857-2949, 2976-2979, 2981, 3000, 3013-3046, 3131-3166, 3184, 3214, 3300-3306, 3354 |
| litebird\_sim/spacecraft.py                      |      111 |       28 |     75% |22, 90-117, 147-203, 302, 307 |
| litebird\_sim/spherical\_harmonics.py            |        4 |        0 |    100% |           |
| litebird\_sim/units.py                           |       51 |       10 |     80% |56, 61, 73, 99, 133, 136, 158-169 |
| litebird\_sim/utilities.py                       |       44 |       10 |     77% |11, 23, 60, 81-87 |
| litebird\_sim/version.py                         |        2 |        0 |    100% |           |
| **TOTAL**                                        | **9369** | **2870** | **69%** |           |


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