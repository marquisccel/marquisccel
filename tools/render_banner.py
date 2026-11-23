# -*- coding: utf-8 -*-
"""Render the room banner as a lit scene instead of hand-coloured rectangles.

The scene is built as buffers (albedo, emission, solidity, surface slope) on a
500x160 logical grid. Light from the window, the two screens and the lamp is
integrated per pixel with distance falloff, then ambient occlusion, bloom and a
filmic curve are applied, and the result is ordered-dithered down to a small
palette. That is what gives it depth: the shading is computed, not painted.
"""
import base64
import io as _io
import numpy as np
from PIL import Image
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           binary_erosion, binary_dilation)

LW, LH = 500, 160
SCALE = 2

albedo = np.zeros((LH, LW, 3), np.float32)
emis = np.zeros((LH, LW, 3), np.float32)
solid = np.zeros((LH, LW), np.float32)
upface = np.zeros((LH, LW), np.float32)      # 1 where a surface faces up
objid = np.zeros((LH, LW), np.int8)          # which object owns each pixel now

YY, XX = np.mgrid[0:LH, 0:LW]


def hexf(c):
    c = c.lstrip('#')
    return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def put(mask, color, e=None, is_solid=True, up=0.0, grad=None, oid=0):
    m = mask.astype(bool)
    col = hexf(color)[None, :] * np.ones((m.sum(), 1), np.float32)
    if grad is not None:
        col = col * grad[m][:, None]
    albedo[m] = col
    if e is not None:
        emis[m] = hexf(e)
    else:
        emis[m] = 0.0
    solid[m] = 1.0 if is_solid else 0.0
    upface[m] = up
    objid[m] = oid


def box(x, y, w, h):
    return (XX >= x) & (XX < x + w) & (YY >= y) & (YY < y + h)


def ellipse(cx, cy, rx, ry):
    return ((XX - cx) / float(rx)) ** 2 + ((YY - cy) / float(ry)) ** 2 <= 1.0


def rrect(x, y, w, h, r):
    inner = box(x + r, y, w - 2 * r, h) | box(x, y + r, w, h - 2 * r)
    for cx, cy in ((x + r, y + r), (x + w - r - 1, y + r),
                   (x + r, y + h - r - 1), (x + w - r - 1, y + h - r - 1)):
        inner |= ellipse(cx, cy, r, r)
    return inner


def vgrad(y0, y1, a, b):
    t = np.clip((YY - y0) / float(max(1, y1 - y0)), 0, 1)
    return (a + (b - a) * t).astype(np.float32)


# ===================================================================== layout
FLOOR_Y = 132
DESK_Y = 92
WX0, WX1, WY0, WY1 = 216, 388, 12, 84
CORNER = 180
PERSON = 303

# ------------------------------------------------------------------- shell
put(box(0, 0, LW, LH), '#243747')                                   # back wall
put(box(0, 0, CORNER, FLOOR_Y), '#1B2A38')                          # near wall
put(box(CORNER, 0, 2, FLOOR_Y), '#3A5970')                          # corner edge
put(box(0, FLOOR_Y - 3, LW, 3), '#31495C')                          # skirting
put(box(0, FLOOR_Y, LW, LH - FLOOR_Y), '#1A2733',
    grad=vgrad(FLOOR_Y, LH, 1.12, 0.78), up=1.0)                    # floor

# door on the near wall
put(box(4, 10, 54, FLOOR_Y - 10), '#16222E')
put(box(56, 10, 3, FLOOR_Y - 10), '#38566C')
put(box(4, 10, 54, 3), '#38566C')
put(box(12, 22, 38, 38), '#121C26')
put(box(12, 68, 38, 44), '#121C26')
put(ellipse(52, 74, 3, 3), '#6E destino'.replace(' destino', '8CA0'))

# ------------------------------------------------------------------ window
put(box(WX0 - 6, WY0 - 6, (WX1 - WX0) + 12, (WY1 - WY0) + 12), '#44687C')
put(box(WX0 - 6, WY0 - 6, (WX1 - WX0) + 12, 3), '#6E96AC')
glass = box(WX0, WY0, WX1 - WX0, WY1 - WY0)
put(glass, '#A8C4CE', e='#20303A', is_solid=False,
    grad=vgrad(WY0, WY1, 1.18, 0.62), oid=2)
# what is outside: a far ridge of buildings and trees
for bx, bw, bh in ((222, 22, 26), (248, 14, 38), (266, 26, 20), (298, 16, 34),
                   (320, 28, 24), (352, 14, 30), (370, 16, 18)):
    put(box(bx, WY1 - 10 - bh, bw, bh) & glass, '#3C5564', is_solid=False, oid=2)
    put(box(bx, WY1 - 10 - bh, bw, 2) & glass, '#51707F', is_solid=False, oid=2)
put(box(WX0, WY1 - 12, WX1 - WX0, 12) & glass, '#2E4452', is_solid=False, oid=2)
for mx in (268, 326):                                               # mullions
    put(box(mx, WY0, 4, WY1 - WY0), '#44687C')
    put(box(mx, WY0, 2, WY1 - WY0), '#6E96AC')
for my in (36, 60):
    put(box(WX0, my, WX1 - WX0, 4), '#44687C')
    put(box(WX0, my, WX1 - WX0, 2), '#6E96AC')
put(box(WX0 - 10, WY1, (WX1 - WX0) + 20, 6), '#5A7E92', up=1.0)
put(box(WX0 - 10, WY1, (WX1 - WX0) + 20, 2), '#84A7B8', up=1.0)

# ------------------------------------------ a small print, left of the window
px, py, pw, ph = 186, 28, 26, 30
put(box(px, py, pw, ph), '#101A24')
put(box(px + 2, py + 2, pw - 4, ph - 4), '#8A4A3A')
put(box(px + 2, py + 13, pw - 4, 7), '#C9743C')
put(box(px + 2, py + 20, pw - 4, 5), '#EDB071')
put(ellipse(px + pw // 2, py + 11, 5, 5), '#F6D28C', e='#2A1D0E')
put(box(px + 2, py + 25, pw - 4, ph - 27), '#2A1E2E')
put(box(px + pw // 2 - 1, py - 3, 2, 3), '#3A5970')

# ------------------------------------------------- shelves, dresser, plant
for sy in (34, 58):
    put(box(400, sy, 96, 4), '#6B5038', up=1.0)
    put(box(400, sy, 96, 2), '#8A6A4A', up=1.0)
    bx, i = 404, 0
    while bx < 492:
        bh = 10 + (i * 7) % 12
        put(box(bx, sy - bh, 5, bh), ['#3A4C68', '#4C3A50', '#32526E', '#5A452C',
                                      '#2A4258', '#44506C'][i % 6])
        put(box(bx, sy - bh, 5, 2), '#7084A0')
        bx += 6
        i += 1

put(box(452, 88, 48, FLOOR_Y - 88), '#5A4330', grad=vgrad(88, FLOOR_Y, 1.1, 0.8))
put(box(452, 88, 48, 3), '#7E6145', up=1.0)
for dy in (94, 104, 114, 124):
    put(box(456, dy, 40, 8), '#4A3526')
    put(box(456, dy, 40, 2), '#6E5338')
    put(box(472, dy + 3, 8, 2), '#9C7C58')
put(box(458, 80, 18, 8), '#16222E')
put(box(460, 82, 14, 4), '#1E3040')

# books on the floor
for i, bw in enumerate((30, 27, 32, 25, 29)):
    put(box(404, FLOOR_Y - 5 - i * 5, bw, 5), ['#5A452C', '#2A4258', '#4C3A50',
                                               '#32526E', '#3A4C68'][i % 5])
    put(box(404, FLOOR_Y - 5 - i * 5, bw, 2), '#7084A0', up=1.0)

# --------------------------------------------------------------- the desk
put(box(200, DESK_Y, 196, 5), '#B9CBD8', up=1.0)
put(box(200, DESK_Y, 196, 2), '#DCE7EF', up=1.0)
put(box(200, DESK_Y + 5, 196, 4), '#7E919F')
put(box(206, DESK_Y + 9, 16, FLOOR_Y - DESK_Y - 9), '#56687A')
put(box(206, DESK_Y + 9, 4, FLOOR_Y - DESK_Y - 9), '#7E919F')
put(box(374, DESK_Y + 9, 16, FLOOR_Y - DESK_Y - 9), '#56687A')
put(box(374, DESK_Y + 9, 4, FLOOR_Y - DESK_Y - 9), '#7E919F')


def monitor(x0, w, kind):
    top = 58
    h = DESK_Y - top
    put(rrect(x0, top, w, h, 3), '#121C26')
    put(box(x0 + 1, top + 1, w - 2, 2), '#2E4660')
    sx, sy, sw, sh = x0 + 4, top + 4, w - 8, h - 11
    put(box(sx, sy, sw, sh), '#16263C', e='#0E1A2A', is_solid=False)
    rng = np.random.default_rng(7 if kind == 'edit' else 11)
    if kind == 'edit':
        put(box(sx, sy, 10, sh), '#1B2F48', e='#101D2E', is_solid=False)
        put(box(sx, sy, sw, 3), '#223C5E', e='#13243A', is_solid=False)
        y = sy + 5
        while y < sy + sh - 3:
            cx = sx + 13 + int(rng.integers(0, 3)) * 3
            for _ in range(int(rng.integers(1, 4))):
                lw = int(rng.integers(8, 26))
                if cx + lw > sx + sw - 2:
                    break
                c = ['#8FB4DC', '#D7E6F6', '#5E8CBE', '#D97FBF', '#6FC3A0',
                     '#E9A94F', '#8FB4DC', '#5E8CBE'][int(rng.integers(0, 8))]
                put(box(cx, y, lw, 2), c, e=c, is_solid=False)
                cx += lw + 3
            put(box(sx + 3, y, 2, 2), '#3B5E8C', e='#1E3049', is_solid=False)
            y += 4
    else:
        put(box(sx, sy, sw, 3), '#223C5E', e='#13243A', is_solid=False)
        y = sy + 5
        while y < sy + sh - 3:
            cx = sx + 4
            if rng.random() < .55:
                put(box(cx, y, 2, 2), '#5FB0E0', e='#2E5A74', is_solid=False)
                cx += 5
            lw = int(rng.integers(14, sw - 10))
            c = '#6FC3A0' if rng.random() < .3 else '#B9CCDF'
            put(box(cx, y, min(lw, sx + sw - 2 - cx), 2), c, e=c, is_solid=False)
            y += 4
    put(box(x0 + w // 2 - 5, DESK_Y - 7, 10, 7), '#16222E')
    put(box(x0 + w // 2 - 16, DESK_Y - 2, 32, 2), '#2E4660', up=1.0)
    return (sx, sy, sw, sh)


SA = monitor(222, 70, 'edit')
SB = monitor(314, 70, 'term')

# keyboard, mouse
put(rrect(256, DESK_Y - 4, 64, 4, 1), '#223040')
for k in range(258, 318, 6):
    put(box(k, DESK_Y - 4, 4, 2), '#3A5168')
put(ellipse(332, DESK_Y - 2, 5, 3), '#3A5168')

# coffee, right end of the desk
CUX, CUY = 350, DESK_Y - 15
put(ellipse(CUX + 9, DESK_Y - 1, 16, 4), '#B9CBD8', up=1.0)
put(ellipse(CUX + 9, DESK_Y - 2, 12, 3), '#DCE7EF', up=1.0)
handle = ellipse(CUX + 20, CUY + 7, 6, 6) & ~ellipse(CUX + 20, CUY + 7, 3, 3)
put(handle & box(CUX + 17, 0, 20, LH), '#E4ECF2')
cup = (box(CUX, CUY, 19, 13) & ~ellipse(CUX + 9, CUY + 13, 11, 4))
cup = cup | (ellipse(CUX + 9, CUY + 11, 9, 4) & box(0, CUY + 11, LW, 6))
put(cup, '#EEF4F8', grad=vgrad(CUY, CUY + 14, 1.0, 0.78))
put(box(CUX, CUY, 19, 2), '#FFFFFF')
put(box(CUX + 2, CUY + 2, 3, 9), '#FFFFFF')
put(ellipse(CUX + 9, CUY + 1, 8, 2), '#4A3324', e='#1A1008')

# ------------------------------------------------------- chair and person
CH_X, CH_W = 276, 54

# the person: skull, jaw, neck, shoulders
head = ellipse(PERSON, 56, 12, 12) | box(PERSON - 9, 56, 18, 8)
head |= ellipse(PERSON, 62, 9, 7)
put(head, '#0D1119', oid=1)
put(box(PERSON - 5, 68, 10, 6), '#241C38', oid=1)                       # neck
sh = (ellipse(PERSON, 104, 27, 36) & box(0, 72, LW, 60)) | box(PERSON - 26, 86, 52, 30)
sh |= ellipse(PERSON - 20, 84, 10, 11) | ellipse(PERSON + 20, 84, 10, 11)
put(sh, '#453872', grad=vgrad(72, 116, 1.18, 0.78), oid=1)

# chair back: a narrower lumbar panel so the shoulders read either side
cb = rrect(CH_X, 80, CH_W, 40, 8)
put(cb, '#121C27', grad=vgrad(80, 120, 1.3, 0.85))
put(rrect(CH_X + 4, 84, CH_W - 8, 32, 6), '#0A121B')
for qy in range(88, 112, 6):
    put(rrect(CH_X + 7, qy, CH_W - 14, 2, 1), '#1D2B3A')
put(box(CH_X, 84, 2, 32), '#1E2C3C')
put(box(CH_X + CH_W - 2, 84, 2, 32), '#1E2C3C')
put(rrect(CH_X - 13, 96, 15, 5, 2), '#17222E')
put(rrect(CH_X + CH_W - 2, 96, 15, 5, 2), '#17222E')

# legs below the seat, then the seat and base in front of them
for side in (-1, 1):
    lx = PERSON + side * 28 - 5
    put(rrect(lx, 118, 11, 24, 3), '#2B3650', grad=vgrad(118, 142, 1.12, 0.82))
    put(rrect(lx - 4, 139, 19, 8, 3), '#1A2130')
    put(box(lx - 4, 139, 19, 2), '#2C3647')

put(rrect(CH_X - 6, 118, CH_W + 12, 10, 4), '#17222E', up=1.0)
put(box(CH_X - 6, 118, CH_W + 12, 2), '#253749', up=1.0)
put(box(PERSON - 7, 127, 14, 11), '#0E1822')
put(rrect(CH_X + 6, 136, CH_W - 12, 5, 2), '#0E1822', up=1.0)
put(rrect(CH_X + 3, 140, 13, 4, 2), '#0A121B')
put(rrect(CH_X + CH_W - 16, 140, 13, 4, 2), '#0A121B')

# ------------------------------------------------------------ rug and cat
put(rrect(392, 138, 100, 20, 6), '#2D3A48', up=1.0,
    grad=vgrad(138, 158, 1.1, 0.88))
put(rrect(396, 142, 92, 2, 1), '#3A4A5C', up=1.0)
put(rrect(396, 152, 92, 2, 1), '#3A4A5C', up=1.0)
cat_body = ellipse(442, 148, 22, 8)
put(cat_body, '#6E7B88', grad=vgrad(140, 156, 1.15, 0.85))
put(ellipse(442, 143, 20, 3), '#95A2AE')
put(ellipse(422, 145, 8, 7), '#6E7B88')
put(ellipse(422, 142, 7, 3), '#95A2AE')
put(box(417, 137, 3, 5), '#6E7B88')
put(box(424, 137, 3, 5), '#6E7B88')
put(box(418, 145, 5, 1), '#2C353F')
put(ellipse(462, 152, 9, 2), '#6E7B88')

# ======================================================== lighting solution
amb = np.array([0.33, 0.365, 0.42], np.float32)
light = np.zeros((LH, LW, 3), np.float32)


def add_light(mask, color, strength, falloff):
    d = distance_transform_edt(~mask.astype(bool))
    att = np.exp(-d / float(falloff)).astype(np.float32)
    light[:] += att[:, :, None] * hexf(color)[None, None, :] * strength


add_light(glass, '#DCEAF2', 1.15, 76)                     # the window
add_light(box(SA[0], SA[1], SA[2], SA[3]), '#A8CEEE', 0.70, 30)
add_light(box(SB[0], SB[1], SB[2], SB[3]), '#A8CEEE', 0.70, 30)
add_light(box(458, 80, 18, 8), '#E9A94F', 0.45, 22)       # the clock
add_light(box(px + 2, py + 2, pw - 4, ph - 4), '#D98A4A', 0.28, 18)

# surfaces facing up catch more of the window
light += (upface * 0.18)[:, :, None] * hexf('#BFD9E6')[None, None, :]

# ambient occlusion from how enclosed each pixel is
occ = gaussian_filter(solid, 3.0)
ao = np.clip(1.0 - 0.55 * occ, 0.35, 1.0)
contact = gaussian_filter(solid * (upface < 0.5), 1.2)
ao *= np.clip(1.0 - 0.3 * contact, 0.55, 1.0)

shaded = albedo * (amb[None, None, :] + light) * ao[:, :, None]

# filmic curve on the lit pass, then emission on top so screens stay readable
shaded = shaded / (1.0 + shaded) * 1.62
shaded = shaded + emis * 0.70

# the window is the one thing that must stay readable, so its values are set
# directly rather than being pushed through the curve with everything else
gy = (YY - WY0) / float(WY1 - WY0)
sky = 0.86 - 0.26 * np.clip(gy, 0, 1)
winv = np.where(glass, sky, 0).astype(np.float32)
far = np.zeros((LH, LW), np.float32)
for bx, bw, bh in ((222, 22, 26), (248, 14, 38), (266, 26, 20), (298, 16, 34),
                   (320, 28, 24), (352, 14, 30), (370, 16, 18)):
    m = box(bx, WY1 - 10 - bh, bw, bh) & glass
    far[m] = 0.33
    far[box(bx, WY1 - 10 - bh, bw, 2) & glass] = 0.47
far[box(WX0, WY1 - 12, WX1 - WX0, 12) & glass] = 0.24
winv = np.where(far > 0, far, winv)
gm = objid == 2
shaded[gm] = (winv[gm][:, None] * hexf('#E6F1F7')[None, :])

# a cool rim where the backlit figure meets the window
pm = objid == 1
edge = pm & ~binary_erosion(pm, np.ones((3, 3), bool))
bright_bg = (shaded.mean(axis=2) > 0.40) & ~pm
rim = edge & binary_dilation(bright_bg, np.ones((3, 3), bool))
shaded[rim] = np.clip(shaded[rim] + hexf('#A8CEE4') * 0.42, 0, 1)

# bloom, gently
lum = shaded.mean(axis=2)
hot = np.clip(lum - 0.56, 0, None)
shaded += (gaussian_filter(hot, 6.0) * 0.30)[:, :, None] * hexf('#BBD6E8')
shaded += (gaussian_filter(hot, 18.0) * 0.16)[:, :, None] * hexf('#8FB4D0')

# the left of the frame falls away, so the name has somewhere quiet to sit
ramp = np.clip((XX - 36) / 160.0, 0, 1) ** 0.9
shaded *= (0.42 + 0.58 * ramp)[:, :, None]

# pull some colour back after the tone curve
g = shaded.mean(axis=2, keepdims=True)
shaded = np.clip(g + (shaded - g) * 1.40, 0, 1)
shaded = shaded ** np.array([1.02, 1.0, 0.97], np.float32)[None, None, :]

# ordered dither, so the gradients break up like pixel art rather than banding
B = np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
              [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
              [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
              [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]],
             np.float32) / 64.0 - 0.5
dither = np.tile(B, (LH // 8 + 1, LW // 8 + 1))[:LH, :LW]
shaded = np.clip(shaded + dither[:, :, None] * (7.0 / 255.0), 0, 1)

img = Image.fromarray((shaded * 255).astype(np.uint8), 'RGB')
img = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=52, dither=Image.Dither.NONE)
img = img.convert('RGB').resize((LW * SCALE, LH * SCALE), Image.Resampling.NEAREST)
img = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=52, dither=Image.Dither.NONE)

buf = _io.BytesIO()
img.save(buf, format='PNG', optimize=True)
png = buf.getvalue()
open('assets/room.png', 'wb').write(png)
print('room.png %.1f KB' % (len(png) / 1024.0))

b64 = base64.b64encode(png).decode('ascii')

# =============================================================== svg wrapper
# The raster carries the lighting; the name stays vector so it is crisp at any
# size, and a few vector accents keep the scene alive without re-rendering.
K = SCALE
FW, FH = LW * K, LH * K
gx, gy, gw = WX0 * K, WY0 * K, (WX1 - WX0) * K
ax, ay, aw, ah = [v * K for v in SA]
cux, cuy = CUX * K, CUY * K
clx, cly = 458 * K, 80 * K
rain_h = ay - gy
SANS = "'Segoe UI', Arial, Helvetica, sans-serif"
MONO = "Consolas, 'Liberation Mono', monospace"

svg = """<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Ega Yurcel Satriaji &#8212; Software Engineer</title>
<desc id="d">Pixel art of a room at night. Rain runs down a tall window that lights the desk; two monitors show code, a cup of coffee steams at the right of the desk, and someone works in the chair seen from behind.</desc>
<style>
.rain{{animation:rn 2.1s linear infinite}}
@keyframes rn{{from{{transform:translate(0,0)}}to{{transform:translate(-18px,48px)}}}}
.cur{{animation:cur 1.1s steps(1,end) infinite}}
@keyframes cur{{50%{{opacity:0}}}}
.steam{{animation:st 5.6s linear infinite;opacity:0}}
@keyframes st{{0%{{opacity:0;transform:translate(0,0)}}22%{{opacity:.55}}
100%{{opacity:0;transform:translate(7px,-30px)}}}}
.blink{{animation:bl 2s steps(2,end) infinite}}
@keyframes bl{{0%,50%{{opacity:1}}51%,100%{{opacity:.15}}}}
@media (prefers-reduced-motion:reduce){{.rain,.cur,.steam,.blink{{animation:none}}}}
</style>
<defs>
<clipPath id="fr"><rect width="{W}" height="{H}" rx="10"/></clipPath>
<clipPath id="win"><rect x="{gx}" y="{gy}" width="{gw}" height="{rain_h}"/></clipPath>
<pattern id="rn" width="18" height="48" patternUnits="userSpaceOnUse">
<path d="M4 1 l-2 10 M13 16 l-2 10 M7 32 l-2 10" stroke="#EAF3F8"
stroke-opacity=".30" stroke-width="1" fill="none"/></pattern>
</defs>
<g clip-path="url(#fr)">
<image x="0" y="0" width="{W}" height="{H}" image-rendering="pixelated"
preserveAspectRatio="none" xlink:href="data:image/png;base64,{b64}"/>
<g clip-path="url(#win)"><g class="rain"><rect x="{gx}" y="{gy}" width="{gw}" height="{rain_h}" fill="url(#rn)"/></g></g>
<rect class="cur" x="{curx}" y="{cury}" width="4" height="4" fill="#EAF3F8"/>
<g class="steam" style="animation-delay:0s"><rect x="{s1x}" y="{s1y}" width="4" height="2" fill="#CFE0EA"/></g>
<g class="steam" style="animation-delay:1.9s"><rect x="{s2x}" y="{s2y}" width="4" height="2" fill="#CFE0EA"/></g>
<g class="steam" style="animation-delay:3.8s"><rect x="{s3x}" y="{s3y}" width="4" height="2" fill="#CFE0EA"/></g>
<rect class="blink" x="{clx1}" y="{cly1}" width="2" height="2" fill="#7FD8EE"/>
<rect class="blink" x="{clx2}" y="{cly1}" width="2" height="2" fill="#7FD8EE"/>
<text x="56" y="62" font-family="{MONO}" font-size="11" fill="#5E8FB4" letter-spacing="4.5">MALANG, INDONESIA</text>
<text x="54" y="120" font-family="{SANS}" font-size="48" font-weight="800" fill="#F4F8FC" letter-spacing="-1">EGA YURCEL</text>
<text x="54" y="164" font-family="{SANS}" font-size="48" font-weight="800" fill="#7FBCEE" letter-spacing="-1">SATRIAJI<tspan fill="#E9A94F">.</tspan></text>
<text x="56" y="198" font-family="{MONO}" font-size="15" fill="#9BB4C8" letter-spacing="2.5">Software Engineer</text>
<rect x="0.5" y="0.5" width="{W1}" height="{H1}" rx="10" fill="none" stroke="#1B2E3E"/>
</g></svg>""".format(
    W=FW, H=FH, W1=FW - 1, H1=FH - 1, b64=b64, SANS=SANS, MONO=MONO,
    gx=gx, gy=gy, gw=gw, rain_h=rain_h,
    curx=ax + 26, cury=ay + ah - 8,
    s1x=cux + 10, s1y=cuy - 10, s2x=cux + 14, s2y=cuy - 16,
    s3x=cux + 8, s3y=cuy - 22,
    clx1=clx + 8, clx2=clx + 20, cly1=cly + 6)

open('assets/banner-pixel.svg', 'w', encoding='utf-8').write(svg)
print('banner %.1f KB' % (len(svg) / 1024.0))
