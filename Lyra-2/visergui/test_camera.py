"""Hand-computed sanity tests for `viser_camera_to_opencv_viewmat`.

Run with:
    python -m pytest visergui/test_camera.py -q
or just:
    python visergui/test_camera.py
"""

from __future__ import annotations

import numpy as np

from viewer import viser_camera_to_opencv_viewmat


def _close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    return np.allclose(a, b, atol=tol, rtol=0.0)


def test_identity_pose_translated():
    """Camera at world (0, 0, 5), identity rotation. Viser uses OpenCV
    conventions internally, so identity c2w means the camera looks +Z_world.
    World origin (0,0,0) is at z=-5 *behind* the camera in its own frame."""
    M = viser_camera_to_opencv_viewmat(
        position=(0.0, 0.0, 5.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    expected = np.array(
        [
            [1.0, 0.0, 0.0,  0.0],
            [0.0, 1.0, 0.0,  0.0],
            [0.0, 0.0, 1.0, -5.0],
            [0.0, 0.0, 0.0,  1.0],
        ],
        dtype=np.float64,
    )
    assert _close(M, expected), f"identity pose viewmat wrong:\n{M}\nexpected\n{expected}"

    origin_cam = M @ np.array([0.0, 0.0, 0.0, 1.0])
    assert _close(origin_cam[:3], np.array([0.0, 0.0, -5.0]))


def test_looking_at_origin_from_plus_z():
    """Camera at (0, 0, 5), rotated 180° about X (wxyz=(0,1,0,0)) so it now
    faces world -Z, looking at the origin with world-up=+Y. R_c2w = diag(1,-1,-1)."""
    M = viser_camera_to_opencv_viewmat(
        position=(0.0, 0.0, 5.0),
        wxyz=(0.0, 1.0, 0.0, 0.0),
    )
    origin_cam = M @ np.array([0.0, 0.0, 0.0, 1.0])
    assert _close(origin_cam[:3], np.array([0.0, 0.0, 5.0])), (
        f"origin should be at +5 along camera-Z (in front), got {origin_cam[:3]}"
    )

    # World +Y is "up"; in OpenCV (y-down) the "above optical axis" pixel is at
    # negative camera-y.
    above_cam = M @ np.array([0.0, 1.0, 0.0, 1.0])
    assert _close(above_cam[:3], np.array([0.0, -1.0, 5.0])), (
        f"world +Y should land at negative camera-y, got {above_cam[:3]}"
    )


def test_orthonormal_rotation_block():
    """For any unit quaternion the 3x3 rotation block must be orthonormal."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        q = rng.normal(size=4)
        q = q / np.linalg.norm(q)
        pos = rng.normal(size=3) * 5.0
        M = viser_camera_to_opencv_viewmat(pos, q)
        R = M[:3, :3]
        I = R @ R.T
        assert _close(I, np.eye(3), tol=1e-9), f"R is not orthonormal:\n{I}"
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-9, f"det(R) = {det}, expected 1"


if __name__ == "__main__":
    test_identity_pose_translated()
    test_looking_at_origin_from_plus_z()
    test_orthonormal_rotation_block()
    print("all camera tests passed")
