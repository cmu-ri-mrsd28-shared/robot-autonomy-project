"""Rotation conversion compatible with ROS Noetic's older SciPy."""


def quaternion_from_matrix(matrix, rotation_class=None):
    """Return an x, y, z, w quaternion from a rotation matrix."""
    if rotation_class is None:
        from scipy.spatial.transform import Rotation
        rotation_class = Rotation
    # Ubuntu Focal ships SciPy 1.3, which called this constructor from_dcm.
    constructor = getattr(rotation_class, 'from_matrix', None)
    if constructor is None:
        constructor = rotation_class.from_dcm
    return constructor(matrix).as_quat()
