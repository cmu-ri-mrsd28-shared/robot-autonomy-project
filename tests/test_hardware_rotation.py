import numpy as np
from catchbox.hardware.rotation import quaternion_from_matrix


def test_uses_legacy_scipy_from_dcm():
    class LegacyRotation:
        @staticmethod
        def from_dcm(matrix):
            np.testing.assert_array_equal(matrix, np.eye(3))
            class Result:
                def as_quat(self):return [0, 0, 0, 1]
            return Result()
    assert quaternion_from_matrix(np.eye(3), LegacyRotation)==[0,0,0,1]


def test_installed_scipy_identity_rotation():
    np.testing.assert_allclose(quaternion_from_matrix(np.eye(3)), [0,0,0,1])
