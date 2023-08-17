# NOTE: Keep curve coefficients constant through testing.
#       We should have already seen that determine_mech_coeffs works fine.
CURVE_COEFFS = (-2.579E-03, 2.311E-02, -2.155E-03, 3.703E-05, -1.367E-06)

test_1 = 11.566, (2, 15, 12), 40, 2.11, 1.225, CURVE_COEFFS
test_2 = 0, (2, 15, 12), 40, 2.11, 1.225, CURVE_COEFFS
test_3 = 13, (2, 15, 12), 40, 2.11, 1.225, CURVE_COEFFS
test_4 = 20, (2, 15, 12), 40, 2.11, 1.225, CURVE_COEFFS

test_cases = (test_1, test_2, test_3, test_4)
__all__ = [test_cases]
