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
