# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                        |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------- | -------: | -------: | ------: | --------: |
| litebird\_sim/\_\_init\_\_.py               |       40 |        0 |    100% |           |
| litebird\_sim/bandpass\_template\_module.py |       73 |       64 |     12% |20-27, 41-45, 64-91, 104, 133-219 |
| litebird\_sim/bandpasses.py                 |      127 |       23 |     82% |101, 117, 129-130, 235-238, 246-247, 256-257, 268, 314-315, 319-332, 347 |
| litebird\_sim/beam\_convolution.py          |      109 |       16 |     85% |130, 148-150, 167, 297, 407-409, 414-416, 430, 447-454 |
| litebird\_sim/beam\_synthesis.py            |      104 |        8 |     92% |249-253, 297, 304-305, 318 |
| litebird\_sim/compress.py                   |       24 |        8 |     67% | 17, 27-33 |
| litebird\_sim/constants.py                  |       16 |        0 |    100% |           |
| litebird\_sim/coordinates.py                |       65 |       22 |     66% |110-114, 146-155, 183-193, 213-231 |
| litebird\_sim/detectors.py                  |      224 |       10 |     96% |17, 213-217, 274, 278-283, 444, 465, 592 |
| litebird\_sim/dipole.py                     |      112 |       42 |     62% |61, 67-69, 75-78, 83-84, 89-94, 99-102, 112-113, 120-125, 139-185, 250, 260, 270-279 |
| litebird\_sim/distribute.py                 |       75 |        9 |     88% |   113-122 |
| litebird\_sim/gaindrifts.py                 |      121 |       12 |     90% |239, 340, 392-396, 450, 468, 474, 549, 552, 556 |
| litebird\_sim/grasp2alm.py                  |      363 |      106 |     71% |88, 139-157, 169-193, 265-283, 299-313, 347, 356, 378-388, 397-420, 484-485, 492-493, 496, 503-504, 520, 527, 554, 558, 588, 602, 615-626, 686, 691, 704, 712, 720, 728, 795, 801-804 |
| litebird\_sim/healpix.py                    |      115 |       37 |     68% |52, 56, 99-100, 104-133, 215, 257, 271, 297 |
| litebird\_sim/hwp.py                        |       98 |       39 |     60% |41, 71, 108, 120, 125, 130, 139-150, 156-177, 214, 296, 310-319, 322-325, 347 |
| litebird\_sim/hwp\_diff\_emiss.py           |       31 |        6 |     81% |15, 20-21, 43-45, 80 |
| litebird\_sim/hwp\_harmonics.py             |      227 |      131 |     42% |44, 48-51, 56-65, 77-105, 122-170, 175-212, 220-222, 229-233, 240-244, 267-277, 307-351, 374-385, 410-456, 542-552, 566-591, 622-641, 655-664, 674, 690-693, 703-707, 731-733, 744-758, 800 |
| litebird\_sim/imo/\_\_init\_\_.py           |        3 |        0 |    100% |           |
| litebird\_sim/imo/imo.py                    |       81 |       21 |     74% |38-53, 58-64, 78, 96, 100, 113, 148 |
| litebird\_sim/imobrowser.py                 |      187 |      141 |     25% |29, 61-70, 75-83, 90-106, 109-112, 115-118, 121-124, 127-131, 134-138, 141-157, 160-174, 177-178, 181, 186-233, 236, 239, 242, 245-250, 253-256, 259-263, 267, 272-307, 310-313, 316-322, 325-331, 334, 338-343, 347-365 |
| litebird\_sim/input\_sky.py                 |      196 |       18 |     91% |99, 103, 109, 137-140, 145, 213, 238, 240, 259, 280, 320, 357, 413, 482, 514-515, 522 |
| litebird\_sim/install\_imo.py               |      114 |       95 |     17% |23-32, 36-49, 59, 69-172, 181-233, 237-246, 255-273, 277 |
| litebird\_sim/io.py                         |      259 |       26 |     90% |66, 70, 227-228, 235, 239, 247, 273-274, 297-299, 478, 481, 504-505, 533, 536, 596, 626, 634, 659-661, 756-758, 806 |
| litebird\_sim/madam.py                      |      153 |       11 |     93% |295, 326-329, 349-351, 376, 398, 422, 508 |
| litebird\_sim/mapmaking/\_\_init\_\_.py     |        5 |        0 |    100% |           |
| litebird\_sim/mapmaking/binner.py           |      111 |       44 |     60% |82-90, 116-156, 167-177, 266, 424 |
| litebird\_sim/mapmaking/brahmap\_gls.py     |       13 |        0 |    100% |           |
| litebird\_sim/mapmaking/common.py           |      236 |      149 |     37% |105, 175-176, 191-210, 226-235, 271-276, 299-308, 330-396, 410-421, 425-426, 433-434, 444-448, 454-458, 470-472, 482-507, 511-520, 524-535 |
| litebird\_sim/mapmaking/destriper.py        |      562 |      172 |     69% |112-163, 201, 378-405, 417-429, 525-530, 539-541, 567-591, 618-642, 655-670, 766-770, 797-824, 853-880, 995, 1012, 1134-1140, 1330-1331, 1333-1334, 1374-1384, 1657, 1718-1724, 1736-1738, 1792-1798, 2119, 2131, 2150, 2225, 2235 |
| litebird\_sim/maps\_and\_harmonics.py       |      886 |      284 |     68% |115, 122, 130, 161, 214, 250-257, 279, 356-364, 381, 399-403, 449, 474, 477, 482, 487, 494, 513, 516, 521, 526, 535, 539, 544-572, 581-608, 637, 643, 667, 703, 712, 716, 723-747, 754, 809-814, 835-838, 863, 868-871, 874-877, 880-883, 887-890, 894-897, 930-934, 977-978, 1006-1026, 1029, 1046-1049, 1143, 1163, 1170, 1227-1229, 1390, 1403, 1432, 1435, 1440, 1445, 1454, 1458, 1465, 1468, 1473, 1478, 1485, 1503-1530, 1576, 1613, 1635, 1662, 1666-1669, 1672-1675, 1678-1681, 1685-1688, 1692-1695, 1767-1819, 1905, 1916, 1920, 1927, 1935, 1939, 2067, 2075, 2087, 2097, 2101, 2256, 2356, 2388, 2391-2397, 2483-2513, 2577-2581, 2595-2599, 2607, 2613, 2705, 2737-2750, 2773-2808, 2865-2882, 2937-2940, 2948 |
| litebird\_sim/mpi.py                        |       46 |        9 |     80% |9-13, 105-109 |
| litebird\_sim/mueller\_convolver.py         |      168 |       20 |     88% |108, 117-122, 132-137, 155-156, 229, 231, 233, 235, 256, 351, 363 |
| litebird\_sim/noise.py                      |       61 |       12 |     80% |40-50, 145, 148, 151, 154, 254, 257 |
| litebird\_sim/non\_linearity.py             |       43 |       14 |     67% |32-55, 89, 176, 179, 183 |
| litebird\_sim/observations.py               |      369 |      148 |     60% |221-224, 235-236, 255, 262-263, 289, 302-303, 352-353, 357, 441-459, 464, 466, 473, 478, 508, 544-545, 559-701, 728-730, 765-793, 884, 889-893, 1067, 1078, 1157, 1175, 1238-1264 |
| litebird\_sim/plot\_fp.py                   |      185 |      162 |     12% |22-38, 50-66, 74-97, 105-129, 143-158, 166-197, 200-211, 219-222, 225-339, 345-346 |
| litebird\_sim/pointing\_sys.py              |      193 |       31 |     84% |39, 51, 90, 93-98, 112, 115-119, 132-134, 139-144, 483-485, 555, 562-573 |
| litebird\_sim/pointings.py                  |       39 |        4 |     90% |212, 215-218, 231 |
| litebird\_sim/pointings\_in\_obs.py         |       95 |       13 |     86% |107, 130-136, 173, 293-303 |
| litebird\_sim/profiler.py                   |       38 |        1 |     97% |        66 |
| litebird\_sim/quaternions.py                |       21 |        0 |    100% |           |
| litebird\_sim/scan\_map.py                  |      129 |       51 |     60% |27-29, 35-40, 45-48, 54, 61-62, 77, 92-103, 224, 227, 235, 242-243, 283-285, 294-308, 335-344, 488-494, 504, 510 |
| litebird\_sim/scanning.py                   |      189 |       30 |     84% |40, 117-123, 169-190, 210-213, 288-293, 325-326, 490, 539, 614, 703, 816, 829, 932, 977 |
| litebird\_sim/seeding.py                    |      175 |       28 |     84% |39-53, 93, 107, 138, 225, 229, 232, 239, 248, 250, 257, 263, 266, 273-274, 277, 283, 329 |
| litebird\_sim/simulations.py                |      836 |      239 |     71% |97, 103, 115, 123, 222-250, 410, 437, 440, 478-490, 610-611, 617, 659, 680-681, 684, 689, 694, 784, 807, 829, 880-889, 899, 938, 1028-1030, 1056, 1073, 1076, 1081, 1242-1245, 1281, 1349, 1356-1363, 1393, 1437-1439, 1483-1484, 1540, 1688-1696, 1720-1741, 1813-1823, 1872, 1926, 1931, 2015, 2024, 2057-2067, 2094, 2105-2115, 2166-2171, 2198-2264, 2282-2300, 2315-2324, 2354-2446, 2469-2501, 2510-2543, 2672-2678, 2726 |
| litebird\_sim/spacecraft.py                 |      111 |       28 |     75% |22, 90-117, 147-203, 302, 307 |
| litebird\_sim/spherical\_harmonics.py       |        4 |        0 |    100% |           |
| litebird\_sim/units.py                      |       49 |       19 |     61% |54, 59, 71, 97, 130-163 |
| litebird\_sim/version.py                    |        2 |        0 |    100% |           |
| **TOTAL**                                   | **7483** | **2303** | **69%** |           |


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