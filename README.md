# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| litebird\_sim/\_\_init\_\_.py                                       |       39 |        0 |    100% |           |
| litebird\_sim/bandpasses.py                                         |      124 |       17 |     86% |104, 120, 132-133, 238-241, 253-254, 265, 311-312, 323-329, 344 |
| litebird\_sim/beam\_convolution.py                                  |      112 |       15 |     87% |134, 171, 317, 403-405, 413-415, 428, 437, 455-462, 468 |
| litebird\_sim/beam\_synthesis.py                                    |       73 |        1 |     99% |       110 |
| litebird\_sim/compress.py                                           |       24 |        8 |     67% | 19, 29-35 |
| litebird\_sim/constants.py                                          |       11 |        0 |    100% |           |
| litebird\_sim/coordinates.py                                        |       59 |       28 |     53% |69-74, 99-102, 130-134, 165, 193-203, 223-241 |
| litebird\_sim/detectors.py                                          |      215 |        8 |     96% |18, 256, 260-265, 351, 372, 438 |
| litebird\_sim/dipole.py                                             |      113 |       42 |     63% |64, 70-72, 78-81, 86-87, 92-97, 102-105, 115-116, 123-128, 142-188, 390, 400, 410-419 |
| litebird\_sim/distribute.py                                         |       76 |        9 |     88% |   116-125 |
| litebird\_sim/gaindrifts.py                                         |      118 |       11 |     91% |239, 340, 392-396, 450, 466, 472, 547, 550, 554 |
| litebird\_sim/healpix.py                                            |      122 |       37 |     70% |102, 106, 149-150, 154-183, 265, 307, 321, 344 |
| litebird\_sim/hwp.py                                                |       72 |       25 |     65% |40, 70, 107, 119, 124, 129, 138-144, 150-171, 209, 275 |
| litebird\_sim/hwp\_diff\_emiss.py                                   |       31 |        5 |     84% |18, 23-24, 44, 80 |
| litebird\_sim/hwp\_sys/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| litebird\_sim/hwp\_sys/bandpass\_template\_module.py                |       73 |       63 |     14% |21-28, 42-46, 65-92, 105, 134-218 |
| litebird\_sim/hwp\_sys/examples/script\_non\_linearity\_coupling.py |       62 |       54 |     13% |10-153, 161 |
| litebird\_sim/hwp\_sys/hwp\_sys.py                                  |      244 |      132 |     46% |22, 26-29, 48-57, 69-97, 114-162, 167-204, 212-214, 221-225, 232-236, 259-269, 295-337, 360-371, 394-440, 494-508, 513, 520-523, 549, 560, 565, 568-571, 643, 651, 659-708, 759-763, 767, 821, 836-864 |
| litebird\_sim/imo/\_\_init\_\_.py                                   |        3 |        0 |    100% |           |
| litebird\_sim/imo/imo.py                                            |       69 |       17 |     75% |36-47, 52-57, 71, 89, 93, 106, 110, 141 |
| litebird\_sim/imobrowser.py                                         |      185 |      139 |     25% |30, 62-71, 76-84, 91-107, 110-113, 116-119, 122-125, 128-131, 134-137, 140-156, 159-169, 172-173, 176, 181-228, 231, 234, 237, 240-245, 248-251, 254-258, 262, 267-302, 305-308, 311-317, 320-326, 329, 333-338, 342-360 |
| litebird\_sim/install\_imo.py                                       |      114 |       95 |     17% |24-33, 37-50, 60, 70-173, 182-234, 238-247, 256-274, 278 |
| litebird\_sim/io.py                                                 |      241 |       23 |     90% |67, 71, 227-228, 235, 239, 273-274, 469-475, 478, 501-502, 530, 533, 599, 607, 713-715, 758 |
| litebird\_sim/madam.py                                              |      149 |       11 |     93% |297, 328-331, 351-353, 378, 400, 424, 503 |
| litebird\_sim/mapmaking/\_\_init\_\_.py                             |        4 |        0 |    100% |           |
| litebird\_sim/mapmaking/binner.py                                   |      108 |       44 |     59% |81-89, 115-155, 166-176, 264, 421 |
| litebird\_sim/mapmaking/common.py                                   |      236 |      149 |     37% |105, 175-176, 191-210, 226-235, 271-276, 299-308, 330-396, 410-421, 425-426, 433-434, 444-448, 454-458, 470-472, 482-507, 511-520, 524-535 |
| litebird\_sim/mapmaking/destriper.py                                |      576 |      173 |     70% |112-163, 201, 378-405, 417-429, 457, 527-532, 541-543, 569-593, 620-644, 657-672, 767-771, 798-825, 854-881, 995, 1012, 1134-1140, 1330-1331, 1333-1334, 1374-1384, 1662, 1723-1729, 1741-1743, 1797-1803, 2120, 2132, 2151, 2226, 2236 |
| litebird\_sim/mbs/\_\_init\_\_.py                                   |        1 |        0 |    100% |           |
| litebird\_sim/mbs/mbs.py                                            |      556 |      160 |     71% |40-41, 51-52, 231, 235, 330, 347-348, 364-365, 394-450, 474-476, 482, 486, 492, 510, 513-538, 546-551, 579-581, 590, 607, 614, 629-636, 680-685, 710-711, 717, 722-725, 730, 737, 784-789, 800, 817-818, 824, 830-833, 868-873, 882, 885-951, 979-980, 1017-1018, 1045 |
| litebird\_sim/mpi.py                                                |       46 |        9 |     80% |11-15, 107-111 |
| litebird\_sim/mueller\_convolver.py                                 |      171 |       20 |     88% |99, 108-113, 123-128, 146-147, 220, 222, 224, 226, 247, 342, 354 |
| litebird\_sim/noise.py                                              |       61 |       12 |     80% |43-53, 148, 151, 154, 157, 256, 259 |
| litebird\_sim/non\_linearity.py                                     |       43 |       14 |     67% |35-58, 92, 178, 181, 185 |
| litebird\_sim/observations.py                                       |      347 |      146 |     58% |151-154, 165-166, 185, 192-193, 218, 231-232, 281-282, 286, 369-384, 389, 391, 398, 403, 429, 465-466, 480-622, 649-651, 686-710, 922, 933, 981-998, 1046-1072 |
| litebird\_sim/plot\_fp.py                                           |      184 |      161 |     12% |24-40, 52-68, 76-99, 107-131, 145-157, 165-196, 199-210, 218-221, 224-338, 344-345 |
| litebird\_sim/pointing\_sys.py                                      |      183 |       29 |     84% |40, 52, 91, 94-99, 113, 116-120, 133-135, 140-145, 478-479, 545, 555-558 |
| litebird\_sim/pointings.py                                          |       27 |        0 |    100% |           |
| litebird\_sim/pointings\_in\_obs.py                                 |       94 |       15 |     84% |58, 80-86, 116-119, 238-248 |
| litebird\_sim/profiler.py                                           |       38 |        1 |     97% |        70 |
| litebird\_sim/quaternions.py                                        |       21 |        0 |    100% |           |
| litebird\_sim/scan\_map.py                                          |      109 |       37 |     66% |24-26, 32-37, 42-45, 51, 58-59, 74, 89-100, 202, 205, 245, 275, 397-399, 408-411, 436 |
| litebird\_sim/scanning.py                                           |      186 |       30 |     84% |42, 119-125, 171-192, 212-215, 290-295, 327-328, 494, 540, 614, 702, 815, 828, 931, 976 |
| litebird\_sim/seeding.py                                            |      175 |       28 |     84% |42-56, 98, 112, 143, 226, 230, 233, 240, 249, 251, 258, 264, 267, 274-275, 278, 284, 330 |
| litebird\_sim/simulations.py                                        |      755 |      227 |     70% |93, 99, 111, 119, 216-243, 403, 430, 433, 475-487, 650, 671-672, 675, 680, 685, 775, 798, 820, 874-880, 890, 926, 1076, 1080, 1141, 1148-1155, 1185, 1226-1228, 1268-1269, 1276, 1327, 1473-1480, 1504-1525, 1577, 1587-1588, 1626, 1669, 1674, 1766, 1775, 1806-1816, 1843, 1854-1864, 1887-1905, 1915-1920, 1946-2010, 2028-2044, 2059-2068, 2097-2187, 2210-2239, 2248-2281, 2373-2374, 2471-2477, 2525 |
| litebird\_sim/spacecraft.py                                         |      110 |       28 |     75% |25, 93-120, 150-206, 305, 311 |
| litebird\_sim/spherical\_harmonics.py                               |       54 |        6 |     89% |68, 103, 197, 215-217 |
| litebird\_sim/version.py                                            |        2 |        0 |    100% |           |
|                                                           **TOTAL** | **6416** | **2029** | **68%** |           |


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