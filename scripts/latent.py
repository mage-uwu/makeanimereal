"""Camera and depth helpers shared by the build: rebuild each key's camera exactly as
scripts/render-page.html placed it, decode its depth map, and lift pixels to 3D points."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def direction(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])


class KeyCamera:
    """Perspective camera on the orbit, looking at the target with +Y up (three.js lookAt)."""

    def __init__(self, yaw, pitch, target, radius, fov, aspect):
        self.target = np.asarray(target, float)
        self.pos = self.target + radius * direction(yaw, pitch)
        self.forward = (self.target - self.pos) / np.linalg.norm(self.target - self.pos)
        self.right = np.cross(self.forward, [0.0, 1.0, 0.0])
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.tan = math.tan(math.radians(fov) / 2)
        self.aspect = aspect
        self.radius = radius

    def lift(self, fx, fy, dist):
        """Frame fractions (x right, y down, 0..1) at view distance `dist` -> world points."""
        nx, ny = fx * 2 - 1, 1 - fy * 2
        return (self.pos + self.forward * dist[..., None]
                + self.right * (nx * dist * self.tan * self.aspect)[..., None]
                + self.up * (ny * dist * self.tan)[..., None])

    def project(self, pts):
        """World points -> frame fractions (x right, y down) and view distance."""
        rel = pts - self.pos
        dist = rel @ self.forward
        fx = (rel @ self.right) / (dist * self.tan * self.aspect) * 0.5 + 0.5
        fy = 0.5 - (rel @ self.up) / (dist * self.tan) * 0.5
        return fx, fy, dist


def read_depth(path, radius, pad):
    """View distance per pixel from a depth map written by render.mjs --depth; NaN off the surface."""
    a = np.asarray(Image.open(path).convert('RGBA')).astype(np.float64) / 255
    # three.js unpackRGBToDepth: dot(rgb, (255/256, 255/256/256, 1/65536))
    zw = a[..., 0] * (255 / 256) + a[..., 1] * (255 / 256 / 256) + a[..., 2] / 65536
    near, far = radius - pad, radius + pad
    zn = zw * 2 - 1
    dist = 2 * near * far / (far + near - zn * (far - near))
    dist[a[..., 3] < 0.5] = np.nan
    return dist


class VantageCamera(KeyCamera):
    """A key camera turned by a learned correction: the vantage a drawing actually depicts.

    yaw/pitch offsets move it on the orbit, roll turns it about its view axis (as three.js
    camera.rotateZ), and scale/tx/ty reframe the image about its centre: x' = s (x - .5) + .5 + tx.
    """

    def __init__(self, yaw, pitch, target, radius, fov, aspect, d_yaw=0.0, d_pitch=0.0, roll=0.0, scale=1.0, tx=0.0, ty=0.0):
        super().__init__(yaw + d_yaw, pitch + d_pitch, target, radius, fov, aspect)
        c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        self.right, self.up = c * self.right + s * self.up, -s * self.right + c * self.up
        self.scale, self.tx, self.ty = scale, tx, ty

    def project(self, pts):
        fx, fy, dist = super().project(pts)
        return self.scale * (fx - 0.5) + 0.5 + self.tx, self.scale * (fy - 0.5) + 0.5 + self.ty, dist

    def lift(self, fx, fy, dist):
        return super().lift((fx - 0.5 - self.tx) / self.scale + 0.5, (fy - 0.5 - self.ty) / self.scale + 0.5, dist)


def solve_vantage(view, cam0, fov, aspect, pts, targets, iters=15, huber=0.01):
    """The vantage a drawing actually depicts.

    pts: surface points visible in the key's render; targets: where the drawing puts each one
    (frame fractions, from the drawing's fit to the render). Gauss-Newton with Huber weights over
    (yaw offset, pitch offset, roll in degrees, log scale, tx, ty).
    """
    p = np.zeros(6)

    def residual(q):
        c = VantageCamera(view['yaw'], view['pitch'], cam0['target'], cam0['radius'], fov, aspect,
                          q[0], q[1], q[2], math.exp(q[3]), q[4], q[5])
        fx, fy, _ = c.project(pts)
        return np.concatenate([fx - targets[:, 0], fy - targets[:, 1]])

    steps = np.array([0.05, 0.05, 0.05, 0.001, 0.0005, 0.0005])
    for _ in range(iters):
        r = residual(p)
        J = np.stack([(residual(p + np.eye(6)[i] * steps[i]) - r) / steps[i] for i in range(6)], 1)
        a = np.abs(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        JW = J * w[:, None]
        delta = np.linalg.solve(JW.T @ J + 1e-6 * np.eye(6), -(JW.T @ r))
        p += delta
        if np.abs(delta).max() < 1e-5:
            break
    r = residual(p)
    return {'d_yaw': float(p[0]), 'd_pitch': float(p[1]), 'roll': float(p[2]), 'scale': float(math.exp(p[3])),
            'tx': float(p[4]), 'ty': float(p[5])}, r
