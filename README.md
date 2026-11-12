# JAL A350-900 実寸AR (unofficial fan model)
Procedurally generated Airbus A350-900 in a Japan Airlines (JA01XJ-style) livery at 1:1 scale (metres, Y-up, origin on the ground under the aircraft centre).
Dimensions from Airbus "A350 Family Facts & Figures": length 66.80 m, span 64.75 m, height 17.05 m, fuselage width 5.96 m.
Geometry and textures are generated from scratch by the scripts in `src/`. Unofficial fan model for personal use; not affiliated with, endorsed by, or sponsored by Japan Airlines or Airbus. "JAL", the Tsurumaru crane mark and "Airbus" are trademarks of their respective owners.

## v3 (iOS placement)
- `*-v3.usdz`: same model + `Preliminary_AnchoringAPI` (plane / horizontal) on the root prim. `*-v2.usdz` kept as fallback.
- `ar-banner.html`: hint banner shown inside AR Quick Look (`#custom=` URL fragment).
- `index.html`: pre-AR guidance step before activating AR.
