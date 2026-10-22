#!/usr/bin/env python3
"""Make the VRM small enough to publish (one file under 15 MB): its thumbnail image, which no render
uses, becomes a 1x1 pixel, and the larger PNG textures are re-encoded as lossless WebP (the same
pixels, fewer bytes). Everything else is copied byte for byte.

    python3 coverage/live/slim_vrm.py coverage/assets/magical_girl_02.vrm coverage/live/model.vrm

It also writes <dst>.png: the same bytes packed into an opaque RGB PNG (3 bytes a pixel, after a
4-byte length), because artifacts serve images but not model files; the page unpacks it.
"""
import io
import os
import json
import struct
import sys

from PIL import Image


def main(src, dst):
    b = open(src, 'rb').read()
    n = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + n])
    bin_start = 20 + n + 8
    blob = b[bin_start:bin_start + struct.unpack('<I', b[20 + n:24 + n])[0]]
    thumb = j['extensions']['VRMC_vrm']['meta'].get('thumbnailImage')
    tiny = io.BytesIO()
    Image.new('RGB', (1, 1), (255, 255, 255)).save(tiny, 'PNG')
    replace = {j['images'][thumb]['bufferView']: tiny.getvalue()} if thumb is not None else {}
    for i, im in enumerate(j['images']):
        bv = j['bufferViews'][im['bufferView']]
        if i == thumb or bv['byteLength'] < 100_000:
            continue
        png = Image.open(io.BytesIO(blob[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']]))
        webp = io.BytesIO()
        png.save(webp, 'WEBP', lossless=True, quality=100, method=6, exact=True)
        if len(webp.getvalue()) < bv['byteLength']:
            replace[im['bufferView']] = webp.getvalue()
            im['mimeType'] = 'image/webp'
    out = bytearray()
    for i, bv in enumerate(j['bufferViews']):
        data = replace.get(i, blob[bv.get('byteOffset', 0):bv.get('byteOffset', 0) + bv['byteLength']])
        out += b'\0' * (-len(out) % 4)
        bv['byteOffset'], bv['byteLength'] = len(out), len(data)
        out += data
    out += b'\0' * (-len(out) % 4)
    j['buffers'][0]['byteLength'] = len(out)
    js = json.dumps(j, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(out)
    with open(dst, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, total))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(out), 0x004E4942) + out)
    print(dst, total // 1024, 'KB')
    data = open(dst, 'rb').read()
    payload = struct.pack('<I', len(data)) + data
    width = 2048
    rows = -(-len(payload) // (3 * width))
    payload += b'\0' * (rows * width * 3 - len(payload))
    Image.frombytes('RGB', (width, rows), payload).save(dst + '.png', compress_level=9)
    print(dst + '.png', os.path.getsize(dst + '.png') // 1024, 'KB')


if __name__ == '__main__':
    main(*sys.argv[1:3])
