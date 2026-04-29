# EnergyQuant — Logo Exports

Brand colors:
- Ink (background dark): `#0d1117`
- Paper (background light): `#f4f1e8`
- Accent (Q / highlight): `#f5d547`

Wordmark font: **Archivo** weight 800 (Google Fonts). Letter-spacing −1.4. The "Q" is colored in the accent.

## Folders

### `01-lattice/` — Primary direction (chosen)
Use these everywhere on the EnergyQuant site.

- `lockup-light.svg` / `lockup-dark.svg` — full lockup (icon tile + wordmark), 720×200 viewBox
- `icon-dark.svg` / `icon-light.svg` — padded square icon tile (rounded corners) — for favicon, app icon
- `icon-transparent-dark.svg` / `icon-transparent-light.svg` — mark only, transparent background
- `png/` — rasterized variants at 32, 64, 128, 256, 512, 1024 px (favicons) plus lockups at 720 / 1440 px

### `02-merit-order/` and `06-candle/` — Icon-only marks
Same set of icon variants (no lockup):
- `icon-dark.svg` / `icon-light.svg`
- `icon-transparent-dark.svg` / `icon-transparent-light.svg`
- `png/` — same size set

## Quick HTML/Claude Code usage

```html
<!-- Favicon (PNG fallback + SVG) -->
<link rel="icon" type="image/svg+xml" href="/exports/01-lattice/icon-dark.svg">
<link rel="icon" type="image/png" sizes="32x32" href="/exports/01-lattice/png/icon-dark-32.png">
<link rel="apple-touch-icon" href="/exports/01-lattice/png/icon-dark-256.png">

<!-- Header lockup (use light on dark BG, dark on light BG) -->
<img src="/exports/01-lattice/lockup-dark.svg" alt="EnergyQuant" height="40">
```

Make sure Archivo is loaded in the host page if you re-use the lockup SVG inline (the SVG references the system font stack as fallback):

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@800&display=swap" rel="stylesheet">
```
