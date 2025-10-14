# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                        |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------- | -------: | -------: | ------: | --------: |
| litebird\_sim/\_\_init\_\_.py               |       38 |        0 |    100% |           |
| litebird\_sim/bandpass\_template\_module.py |       72 |       63 |     12% |20-27, 41-45, 64-91, 104, 133-217 |
| litebird\_sim/bandpasses.py                 |      123 |       21 |     83% |101, 117, 129-130, 235-238, 250-251, 262, 308-309, 313-326, 341 |
| litebird\_sim/beam\_convolution.py          |      109 |       17 |     84% |131, 149-151, 168, 314, 400-402, 410-412, 425, 434, 452-459 |
| litebird\_sim/beam\_synthesis.py            |       78 |        5 |     94% |108, 257-261 |
| litebird\_sim/compress.py                   |       24 |        8 |     67% | 17, 27-33 |
| litebird\_sim/constants.py                  |       11 |        0 |    100% |           |
| litebird\_sim/coordinates.py                |       58 |       28 |     52% |67-72, 97-100, 128-132, 163, 191-201, 221-239 |
| litebird\_sim/detectors.py                  |      222 |       10 |     95% |17, 211-215, 272, 276-281, 441, 462, 589 |
| litebird\_sim/dipole.py                     |      112 |       42 |     62% |61, 67-69, 75-78, 83-84, 89-94, 99-102, 112-113, 120-125, 139-185, 385, 395, 405-414 |
| litebird\_sim/distribute.py                 |       75 |        9 |     88% |   113-122 |
| litebird\_sim/gaindrifts.py                 |      117 |       11 |     91% |238, 339, 391-395, 449, 465, 471, 546, 549, 553 |
| litebird\_sim/grasp2alm.py                  |      363 |      106 |     71% |88, 136-154, 166-190, 262-280, 296-310, 344, 353, 375-385, 394-417, 482-483, 490-491, 494, 501-502, 518, 525, 552, 556, 586, 600, 613-624, 684, 689, 702, 710, 718, 726, 792, 798-801 |
| litebird\_sim/healpix.py                    |      131 |       37 |     72% |123, 127, 170-171, 175-204, 286, 328, 342, 365 |
| litebird\_sim/hwp.py                        |       98 |       39 |     60% |41, 71, 108, 120, 125, 130, 139-150, 156-177, 214, 296, 310-319, 322-325, 347 |
| litebird\_sim/hwp\_diff\_emiss.py           |       30 |        5 |     83% |15, 20-21, 42, 79 |
| litebird\_sim/hwp\_harmonics.py             |      189 |      106 |     44% |32, 36-39, 58-67, 79-107, 124-172, 177-214, 222-224, 231-235, 242-246, 269-279, 309-353, 376-387, 412-458, 547-572, 603-622, 635, 651-654, 661, 692, 737 |
| litebird\_sim/imo/\_\_init\_\_.py           |        3 |        0 |    100% |           |
| litebird\_sim/imo/imo.py                    |       73 |       16 |     78% |37-48, 53-58, 72, 90, 94, 107, 142 |
| litebird\_sim/imobrowser.py                 |      185 |      139 |     25% |29, 61-70, 75-83, 90-106, 109-112, 115-118, 121-124, 127-130, 133-136, 139-155, 158-168, 171-172, 175, 180-227, 230, 233, 236, 239-244, 247-250, 253-257, 261, 266-301, 304-307, 310-316, 319-325, 328, 332-337, 341-359 |
| litebird\_sim/install\_imo.py               |      114 |       95 |     17% |23-32, 36-49, 59, 69-172, 181-233, 237-246, 255-273, 277 |
| litebird\_sim/io.py                         |      241 |       28 |     88% |65, 69, 225-226, 233, 237, 245, 271-272, 364-365, 467-473, 476, 499-500, 528, 531, 597, 605, 711-713, 749-754, 760 |
| litebird\_sim/madam.py                      |      148 |       15 |     90% |160, 294, 325-328, 348-350, 375, 397, 421, 452-456, 500 |
| litebird\_sim/mapmaking/\_\_init\_\_.py     |        5 |        0 |    100% |           |
| litebird\_sim/mapmaking/binner.py           |      107 |       46 |     57% |82-90, 116-156, 167-177, 254-263, 420 |
| litebird\_sim/mapmaking/brahmap\_gls.py     |       13 |        8 |     38% |   111-137 |
| litebird\_sim/mapmaking/common.py           |      236 |      149 |     37% |105, 175-176, 191-210, 226-235, 271-276, 299-308, 330-396, 410-421, 425-426, 433-434, 444-448, 454-458, 470-472, 482-507, 511-520, 524-535 |
| litebird\_sim/mapmaking/destriper.py        |      575 |      178 |     69% |43, 112-163, 201, 378-405, 417-429, 497, 525-530, 539-541, 567-591, 618-642, 655-670, 746-749, 765-769, 796-823, 852-879, 991, 1008, 1132-1138, 1328-1329, 1331-1332, 1372-1382, 1422, 1660, 1691, 1721-1727, 1739-1741, 1795-1801, 2118, 2130, 2149, 2224, 2234 |
| litebird\_sim/mbs/\_\_init\_\_.py           |        1 |        0 |    100% |           |
| litebird\_sim/mbs/mbs.py                    |      557 |      160 |     71% |69-70, 80-81, 260, 264, 359, 376-377, 393-394, 423-479, 503-505, 511, 515, 521, 539, 542-567, 575-580, 608-610, 619, 636, 643, 658-665, 709-714, 739-740, 746, 751-754, 759, 766, 813-818, 829, 846-847, 853, 859-862, 897-902, 911, 914-980, 1008-1009, 1046-1047, 1074 |
| litebird\_sim/mpi.py                        |       46 |       14 |     70% |9-13, 52, 55, 98-104, 107 |
| litebird\_sim/mueller\_convolver.py         |      170 |       20 |     88% |98, 107-112, 122-127, 145-146, 219, 221, 223, 225, 246, 341, 353 |
| litebird\_sim/noise.py                      |       60 |       12 |     80% |40-50, 145, 148, 151, 154, 253, 256 |
| litebird\_sim/non\_linearity.py             |       42 |       14 |     67% |32-55, 89, 175, 178, 182 |
| litebird\_sim/observations.py               |      364 |      144 |     60% |150-153, 164-165, 184, 191-192, 218, 231-232, 281-282, 286, 369-384, 389, 391, 398, 403, 429, 465-466, 480-622, 649-651, 686-710, 801, 806-810, 981, 992, 1071, 1089, 1152-1178 |
| litebird\_sim/plot\_fp.py                   |      184 |      161 |     12% |22-38, 50-66, 74-97, 105-129, 143-155, 163-194, 197-208, 216-219, 222-336, 342-343 |
| litebird\_sim/pointing\_sys.py              |      183 |       29 |     84% |38, 50, 89, 92-97, 111, 114-118, 131-133, 138-143, 476-477, 543, 553-556 |
| litebird\_sim/pointings.py                  |       37 |        3 |     92% |223, 227, 235 |
| litebird\_sim/pointings\_in\_obs.py         |       94 |       13 |     86% |107, 130-136, 173, 293-303 |
| litebird\_sim/profiler.py                   |       38 |        1 |     97% |        66 |
| litebird\_sim/quaternions.py                |       21 |        0 |    100% |           |
| litebird\_sim/scan\_map.py                  |      113 |       38 |     66% |22-24, 30-35, 40-43, 49, 56-57, 72, 87-98, 205, 208, 248, 273-279, 418-420, 429-432 |
| litebird\_sim/scanning.py                   |      186 |       30 |     84% |40, 117-123, 169-190, 210-213, 288-293, 325-326, 492, 538, 612, 700, 813, 826, 929, 974 |
| litebird\_sim/seeding.py                    |      174 |       28 |     84% |39-53, 93, 107, 138, 221, 225, 228, 235, 244, 246, 253, 259, 262, 269-270, 273, 279, 325 |
| litebird\_sim/simulations.py                |      774 |      238 |     69% |93, 99, 111, 119, 216-243, 403, 430, 433, 473-485, 654, 675-676, 679, 684, 689, 779, 802, 824, 875-884, 894, 932, 1082, 1086, 1147, 1154-1161, 1191, 1228-1230, 1278-1279, 1335, 1351-1352, 1432-1436, 1483-1490, 1514-1535, 1598-1599, 1641, 1688, 1693, 1781, 1790, 1823-1833, 1860, 1871-1881, 1932-1937, 1964-2030, 2048-2066, 2081-2090, 2120-2212, 2235-2267, 2276-2309, 2336-2365, 2440-2446, 2494 |
| litebird\_sim/spacecraft.py                 |      109 |       28 |     74% |22, 90-117, 147-203, 302, 307 |
| litebird\_sim/spherical\_harmonics.py       |      147 |       45 |     69% |101, 136, 189, 192, 290, 308-310, 332, 335, 344-353, 365, 370, 375, 399-405, 413, 418-424, 429-436, 439-441, 446, 499-503 |
| litebird\_sim/version.py                    |        2 |        0 |    100% |           |
|                                   **TOTAL** | **6852** | **2159** | **68%** |           |


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