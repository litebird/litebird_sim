# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/litebird/litebird_sim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | ------: | --------: |
| litebird\_sim/\_\_init\_\_.py                    |       45 |        0 |    100% |           |
| litebird\_sim/bandpass\_template\_module.py      |       74 |       40 |     46% |22-29, 43-47, 66-93, 106, 149-157, 173-174, 179-180, 203, 213-220, 228 |
| litebird\_sim/bandpasses.py                      |      126 |       23 |     82% |108, 124, 136-137, 239-242, 250-251, 260-261, 272, 318-319, 323-336, 351 |
| litebird\_sim/beam\_convolution.py               |      116 |       11 |     91% |134-138, 185, 425-427, 432-434, 448, 465 |
| litebird\_sim/beam\_synthesis.py                 |      107 |        8 |     93% |255-259, 304, 311-312, 325 |
| litebird\_sim/compress.py                        |       24 |        8 |     67% | 17, 27-33 |
| litebird\_sim/constants.py                       |       16 |        0 |    100% |           |
| litebird\_sim/coordinates.py                     |       67 |       22 |     67% |116-120, 152-161, 189-199, 219-237 |
| litebird\_sim/detectors.py                       |      223 |        7 |     97% |18, 274-279, 440, 461, 592 |
| litebird\_sim/dipole.py                          |      317 |      137 |     57% |151, 154, 160-170, 306, 312, 318, 324, 330-332, 338-341, 360-384, 389-390, 395-400, 405-408, 418-419, 424-427, 436-437, 444-450, 480-486, 508-515, 542-568, 580-637, 659-680, 759, 787, 797, 911-919, 930 |
| litebird\_sim/distribute.py                      |       75 |        9 |     88% |   113-122 |
| litebird\_sim/flags.py                           |       61 |        8 |     87% |     69-79 |
| litebird\_sim/gaindrifts.py                      |      115 |       10 |     91% |241, 342, 394-398, 452, 470, 476, 551 |
| litebird\_sim/grasp2alm.py                       |      366 |      106 |     71% |97, 148-166, 178-202, 274-292, 308-322, 359, 369, 391-401, 410-433, 497-498, 505-506, 509, 516-517, 533, 540, 567, 571, 601, 615, 628-639, 699, 704, 717, 725, 733, 741, 808, 814-817 |
| litebird\_sim/healpix.py                         |      115 |       37 |     68% |52, 56, 99-100, 104-133, 215, 257, 271, 297 |
| litebird\_sim/hwp.py                             |       89 |       21 |     76% |70, 102, 143, 155, 160, 165, 178-189, 202-205, 218-220, 308 |
| litebird\_sim/hwp\_diff\_emiss.py                |       32 |        5 |     84% |17, 24-25, 51-53 |
| litebird\_sim/hwp\_harmonics/\_\_init\_\_.py     |        0 |        0 |    100% |           |
| litebird\_sim/hwp\_harmonics/common.py           |       19 |       10 |     47% |7-9, 16-20, 27-31, 54-63 |
| litebird\_sim/hwp\_harmonics/hwp\_harmonics.py   |      214 |       86 |     60% |48, 52-55, 60-69, 105, 126-130, 221-231, 240-265, 296-315, 340, 352-361, 369-372, 382-386, 411-420, 442-444, 466-468, 479-494, 500, 597, 602, 658 |
| litebird\_sim/hwp\_harmonics/jones\_methods.py   |       67 |       52 |     22% |14-46, 52-61, 68-75, 106-135, 169-211 |
| litebird\_sim/hwp\_harmonics/mueller\_methods.py |       22 |       12 |     45% | 23, 61-99 |
| litebird\_sim/hwp\_jones\_parameters.py          |       32 |        7 |     78% |50, 74-77, 82-83 |
| litebird\_sim/hwp\_non\_ideal.py                 |       33 |        8 |     76% |81-90, 93-96 |
| litebird\_sim/imo/\_\_init\_\_.py                |        3 |        0 |    100% |           |
| litebird\_sim/imo/imo.py                         |       82 |       21 |     74% |40-55, 60-66, 80, 98, 102, 115, 150 |
| litebird\_sim/imobrowser.py                      |      187 |      140 |     25% |61-70, 75-83, 90-106, 109-112, 115-118, 121-124, 127-131, 134-138, 141-157, 160-174, 177-178, 181, 186-233, 236, 239, 242, 245-250, 253-256, 259-263, 267, 272-307, 310-313, 316-322, 325-331, 334, 338-343, 347-365 |
| litebird\_sim/input\_sky.py                      |      325 |       32 |     90% |80, 84, 90, 113, 247, 251, 258, 282, 289, 295, 303, 310-314, 317, 330-337, 351, 357, 377, 535, 547, 607, 673, 681, 692, 737, 760, 779-780, 788 |
| litebird\_sim/install\_imo.py                    |      114 |       95 |     17% |23-32, 36-49, 59, 69-172, 181-233, 237-246, 255-273, 277 |
| litebird\_sim/io.py                              |      285 |       21 |     93% |144, 148, 313-314, 325, 363-364, 388-390, 566, 589-590, 621, 684, 714, 722, 742-744, 845-847 |
| litebird\_sim/madam.py                           |      155 |       12 |     92% |244, 298, 329-332, 352-354, 379, 401, 425, 511 |
| litebird\_sim/mapmaking/\_\_init\_\_.py          |        8 |        0 |    100% |           |
| litebird\_sim/mapmaking/binner.py                |      126 |       59 |     53% |89-120, 149-185, 196-206, 464 |
| litebird\_sim/mapmaking/brahmap\_gls.py          |       19 |        1 |     95% |       134 |
| litebird\_sim/mapmaking/common.py                |      287 |      153 |     47% |105, 184, 198-199, 219-249, 262-281, 297-306, 342-347, 370-379, 393-394, 416-482, 514-515, 522-523, 559, 572-576, 582-586, 610-635, 639-648, 652-663, 692 |
| litebird\_sim/mapmaking/destriper.py             |      568 |      158 |     72% |118-169, 207, 385-414, 426-433, 533-538, 551-554, 581-606, 634-659, 672-687, 781-786, 815-843, 874-902, 1158-1164, 1363-1364, 1366-1367, 1705, 1776-1778, 1832-1838, 2159, 2171, 2190, 2265, 2275 |
| litebird\_sim/mapmaking/h\_maps.py               |      141 |      102 |     28% |62-65, 106-123, 144-150, 163-179, 185-189, 195-201, 214-229, 281-389, 406-421 |
| litebird\_sim/mapmaking/pair\_differencing.py    |      141 |       42 |     70% |84-92, 123-161, 213, 248-251, 272-273, 292-293, 299-300, 319-321, 346, 517 |
| litebird\_sim/mapmaking/sanepic\_gls.py          |      193 |      171 |     11% |44-47, 55-65, 72-78, 82-94, 128-206, 226-274, 428-510, 598-669, 713-748 |
| litebird\_sim/maps\_and\_harmonics.py            |     1275 |      401 |     69% |130, 138, 146, 157, 165, 170, 195, 197, 219, 233, 299, 314, 326, 365-372, 394, 493, 498, 504-512, 529, 552-572, 575-579, 617, 624-629, 657, 682, 685, 693, 698, 705, 733, 736, 741, 744, 749, 758, 762, 766, 772, 775, 780, 783, 788, 794, 819-852, 881, 887, 912, 954-957, 963, 971-972, 976, 996, 1000, 1014-1016, 1021-1022, 1042, 1047, 1051, 1055, 1060-1064, 1084-1089, 1129, 1145, 1167-1199, 1234-1246, 1272, 1277-1280, 1283-1286, 1289-1292, 1295-1298, 1302-1303, 1307-1310, 1314-1317, 1361-1378, 1399-1412, 1455-1456, 1484-1504, 1507, 1524-1527, 1624, 1633, 1638, 1648, 1668, 1675, 1690, 1697, 1699, 1778, 1784-1785, 1789, 1792, 1924, 1931-1936, 1988, 1996, 2004, 2042, 2045, 2050, 2053, 2058, 2067, 2071, 2075, 2082, 2085, 2090, 2093, 2098, 2105, 2131-2164, 2210, 2251, 2277, 2283-2286, 2314, 2318-2321, 2324-2327, 2330-2333, 2336-2339, 2343-2344, 2348-2351, 2355-2358, 2455, 2468, 2475, 2480, 2494, 2510, 2515-2526, 2639, 2650, 2654, 2661, 2669, 2673, 2862-2867, 2871, 2879, 2891, 2901, 2905, 2914-2917, 2927-2941, 2986-3000, 3168, 3221-3223, 3288, 3320, 3323-3329, 3415-3445, 3503, 3531-3535, 3549-3553, 3561, 3567, 3651, 3658-3661, 3670-3672, 3692-3697, 3706-3713, 3766-3768, 3792-3827, 3884-3901, 3956-3959, 3967 |
| litebird\_sim/mpi.py                             |       39 |       11 |     72% |10-14, 61, 76-79, 83 |
| litebird\_sim/mueller\_convolver.py              |      144 |        5 |     97% |207-208, 321, 376, 390 |
| litebird\_sim/noise.py                           |      182 |       29 |     84% |19-25, 34-40, 70-83, 272-286, 314, 442-444, 716-734, 767 |
| litebird\_sim/non\_linearity.py                  |       45 |       12 |     73% |34-57, 96, 203 |
| litebird\_sim/observation\_utilities.py          |      123 |        8 |     93% |108, 174, 284-297 |
| litebird\_sim/observations.py                    |      447 |      154 |     66% |274, 299, 325-328, 339-340, 369-370, 414, 446, 502-503, 507, 591-609, 614, 616, 660, 696-697, 713-869, 883-910, 933-935, 970-995, 1321, 1332, 1416, 1434, 1480, 1558 |
| litebird\_sim/planets.py                         |      152 |       15 |     90% |250-253, 286, 320, 340, 439-446 |
| litebird\_sim/plot\_fp.py                        |      185 |      162 |     12% |22-38, 50-66, 74-97, 105-129, 143-158, 166-197, 200-211, 219-222, 225-339, 345-346 |
| litebird\_sim/pointing\_sys.py                   |      182 |       24 |     87% |38, 50, 89, 92-97, 111, 114-118, 131-133, 138-143, 560 |
| litebird\_sim/pointings.py                       |      100 |        4 |     96% |57-58, 298, 398 |
| litebird\_sim/profiler.py                        |       38 |        1 |     97% |        66 |
| litebird\_sim/quaternions.py                     |       21 |        0 |    100% |           |
| litebird\_sim/scan\_map.py                       |      116 |       37 |     68% |35, 42-44, 75-110, 226, 229, 237, 244-245, 286-288, 306-308, 312, 360, 492-498, 508, 514 |
| litebird\_sim/scanning.py                        |      211 |       40 |     81% |44, 121-127, 173-201, 221-224, 299-304, 336-337, 574, 649, 785, 823-833, 898, 911, 1013, 1058 |
| litebird\_sim/seeding.py                         |      173 |       28 |     84% |43-57, 97, 111, 142, 227, 231, 234, 241, 250, 252, 259, 265, 268, 275-276, 279, 285, 331 |
| litebird\_sim/shared\_memory.py                  |       67 |        4 |     94% |37, 66, 81, 100 |
| litebird\_sim/simulations.py                     |     1034 |      264 |     74% |127, 133, 145, 153, 268-296, 493, 531-543, 663-664, 670, 712, 733-734, 737, 742, 747, 837, 860, 882, 933-942, 952, 991, 1081-1083, 1109, 1126, 1129, 1134, 1298-1301, 1337, 1391-1392, 1443, 1453, 1468, 1511, 1514, 1535, 1542-1549, 1579, 1669-1670, 1734, 1782, 1828, 1933, 2121-2142, 2211-2221, 2270, 2326, 2331, 2412, 2421, 2454-2464, 2492, 2503-2513, 2608-2613, 2642-2664, 2696, 2698, 2700, 2716-2748, 2790-2860, 2883-2884, 2886, 2930, 2948-2957, 2989-3085, 3114-3117, 3119, 3140, 3153-3186, 3272-3308, 3326, 3357, 3444-3450, 3498 |
| litebird\_sim/spacecraft.py                      |      107 |       24 |     78% |22, 90-117, 147-199, 298, 303 |
| litebird\_sim/spherical\_harmonics.py            |        4 |        0 |    100% |           |
| litebird\_sim/units.py                           |       51 |       10 |     80% |56, 61, 73, 99, 133, 136, 158-169 |
| litebird\_sim/utilities.py                       |       44 |       10 |     77% |11, 23, 60, 81-87 |
| litebird\_sim/version.py                         |        2 |        0 |    100% |           |
| **TOTAL**                                        | **9731** | **2877** | **70%** |           |


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