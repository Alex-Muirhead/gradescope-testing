# NOTE: Keep curve coefficients constant through testing.
#       We should have already seen that determine_mech_coeffs works fine.
curve_coeffs_1 = (-2.579E-03, 2.311E-02, -2.155E-03, 3.703E-05, -1.367E-06)
curve_coeffs_2 = (-6.798E-03, 3.552E-02, -4.583E-03, 1.395E-04)
curve_coeffs_3 = (1.338E-03, 1.604E-02, 0, 0, -6.22E-06)

test_1 = (2, 18, 12), 10, 60, 0.14, 40, 2.3, 1.225, curve_coeffs_1
test_2 = (2, 17, 11), 10, 70, 0.14, 45, 1.8, 1.225, curve_coeffs_2
test_3 = (2, 16, 10.5), 10, 50, 0.14, 35, 2.2, 1.225, curve_coeffs_3

test_cases = (test_1, test_2, test_3)
__all__ = [test_cases]
