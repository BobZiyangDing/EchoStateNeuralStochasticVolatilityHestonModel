"""
Critical values for the three different models specified for the
Zivot-Andrews unit-root test.

Notes
-----
The p-values are generated through Monte Carlo simulation using 100,000
replications and 2000 data points.
"""
from numpy import array

# constant-only model
c = (
    (0.001, -6.78442),
    (0.100, -5.83192),
    (0.200, -5.68139),
    (0.300, -5.58461),
    (0.400, -5.51308),
    (0.500, -5.45043),
    (0.600, -5.39924),
    (0.700, -5.36023),
    (0.800, -5.33219),
    (0.900, -5.30294),
    (1.000, -5.27644),
    (2.500, -5.03340),
    (5.000, -4.81067),
    (7.500, -4.67636),
    (10.000, -4.56618),
    (12.500, -4.48130),
    (15.000, -4.40507),
    (17.500, -4.33947),
    (20.000, -4.28155),
    (22.500, -4.22683),
    (25.000, -4.17830),
    (27.500, -4.13101),
    (30.000, -4.08586),
    (32.500, -4.04455),
    (35.000, -4.00380),
    (37.500, -3.96144),
    (40.000, -3.92078),
    (42.500, -3.88178),
    (45.000, -3.84503),
    (47.500, -3.80549),
    (50.000, -3.77031),
    (52.500, -3.73209),
    (55.000, -3.69600),
    (57.500, -3.65985),
    (60.000, -3.62126),
    (65.000, -3.54580),
    (70.000, -3.46848),
    (75.000, -3.38533),
    (80.000, -3.29112),
    (85.000, -3.17832),
    (90.000, -3.04165),
    (92.500, -2.95146),
    (95.000, -2.83179),
    (96.000, -2.76465),
    (97.000, -2.68624),
    (98.000, -2.57884),
    (99.000, -2.40044),
    (99.900, -1.88932),
)

# trend-only model
t = (
    (0.001, -83.9094),
    (0.100, -13.8837),
    (0.200, -9.13205),
    (0.300, -6.32564),
    (0.400, -5.60803),
    (0.500, -5.38794),
    (0.600, -5.26585),
    (0.700, -5.18734),
    (0.800, -5.12756),
    (0.900, -5.07984),
    (1.000, -5.03421),
    (2.500, -4.65634),
    (5.000, -4.40580),
    (7.500, -4.25214),
    (10.000, -4.13678),
    (12.500, -4.03765),
    (15.000, -3.95185),
    (17.500, -3.87945),
    (20.000, -3.81295),
    (22.500, -3.75273),
    (25.000, -3.69836),
    (27.500, -3.64785),
    (30.000, -3.59819),
    (32.500, -3.55146),
    (35.000, -3.50522),
    (37.500, -3.45987),
    (40.000, -3.41672),
    (42.500, -3.37465),
    (45.000, -3.33394),
    (47.500, -3.29393),
    (50.000, -3.25316),
    (52.500, -3.21244),
    (55.000, -3.17124),
    (57.500, -3.13211),
    (60.000, -3.09204),
    (65.000, -3.01135),
    (70.000, -2.92897),
    (75.000, -2.83614),
    (80.000, -2.73893),
    (85.000, -2.62840),
    (90.000, -2.49611),
    (92.500, -2.41337),
    (95.000, -2.30820),
    (96.000, -2.25797),
    (97.000, -2.19648),
    (98.000, -2.11320),
    (99.000, -1.99138),
    (99.900, -1.67466),
)

# constant + trend model
ct = (
    (0.001, -38.17800),
    (0.100, -6.43107),
    (0.200, -6.07279),
    (0.300, -5.95496),
    (0.400, -5.86254),
    (0.500, -5.77081),
    (0.600, -5.72541),
    (0.700, -5.68406),
    (0.800, -5.65163),
    (0.900, -5.60419),
    (1.000, -5.57556),
    (2.500, -5.29704),
    (5.000, -5.07332),
    (7.500, -4.93003),
    (10.000, -4.82668),
    (12.500, -4.73711),
    (15.000, -4.66020),
    (17.500, -4.58970),
    (20.000, -4.52855),
    (22.500, -4.47100),
    (25.000, -4.42011),
    (27.500, -4.37387),
    (30.000, -4.32705),
    (32.500, -4.28126),
    (35.000, -4.23793),
    (37.500, -4.19822),
    (40.000, -4.15800),
    (42.500, -4.11946),
    (45.000, -4.08064),
    (47.500, -4.04286),
    (50.000, -4.00489),
    (52.500, -3.96837),
    (55.000, -3.93200),
    (57.500, -3.89496),
    (60.000, -3.85577),
    (65.000, -3.77795),
    (70.000, -3.69794),
    (75.000, -3.61852),
    (80.000, -3.52485),
    (85.000, -3.41665),
    (90.000, -3.28527),
    (92.500, -3.19724),
    (95.000, -3.08769),
    (96.000, -3.03088),
    (97.000, -2.96091),
    (98.000, -2.85581),
    (99.000, -2.71015),
    (99.900, -2.28767),
)


za_critical_values = {"ct": array(ct), "t": array(t), "c": array(c)}

__all__ = ["za_critical_values"]
