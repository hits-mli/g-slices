# G-SLiCEs logo assets

Files are named by the **background they're designed for**:

- `*-light.*` — for **light** backgrounds (dark ink, mid-range viridis)
- `*-dark.*`  — for **dark** backgrounds (light ink, full viridis)

SVG is the source of truth; PNGs are exported (icons 1024², horizontal 1960×600, social 1280×640).
Marks and horizontal lockups have **transparent** backgrounds; tiles and social cards bake in their background.

## Contents

```
icon/        gslices-icon-{light,dark}     square mark, transparent
             gslices-tile-{light,dark}     rounded tile, for avatars / org picture
horizontal/  gslices-horizontal-{light,dark}   mark + title + tagline + method terms (README header)
social/      gslices-social-{light,dark}   1280×640 GitHub social-preview card
```

## README header (auto light/dark)

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logos/horizontal/gslices-horizontal-dark.svg">
  <img alt="G-SLiCEs" src="assets/logos/horizontal/gslices-horizontal-light.svg" width="560">
</picture>
```

(Adjust the path to wherever you place this folder in the repo.)

## GitHub social preview

Repo → Settings → General → Social preview → upload `social/gslices-social-dark.png` (1280×640).

## Org avatar (hits-mli)

Org → Settings → Profile → Profile picture → `icon/gslices-tile-dark.png` (or `-light`).

## Palette

viridis — perceptually uniform, colorblind-safe (deuteranopia/protanopia), and legible in grayscale.
Yellow = observations · blue→teal = observed/latent series · teal→green→yellow = forecast densities (color also encodes lead time).
