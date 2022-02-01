# clrays

## Features
- [x] camera controls
- [x] custom keybindings
- [x] skycolour, skybox: sphere
- [x] export frame
- [x] mesh: triangle meshes (.obj)
- [x] BVH: binning + SAH + top-level
- [ ] BVH: 4 way

### GPU
- [x] basic pathtracer (area lights, materials, speculars, dielectrics, beer's law)
- [x] frame energy
- [x] tone mapping: Aces, Hable/Uncharted
- [ ] microfacet materials
  - [x] GGX-Smith conductor
  - [ ] GGX-Smith dielectric
  - [x] GGX NDF importance sampling
  - [ ] GGX VNDF importance sampling
- [ ] utilize BVH

### CPU
- [x] primitives: planes, spheres, triangles
- [x] material
- [x] blinn shading
- [x] reflection
- [x] refraction
- [x] absorption
- [x] multithreading
- [x] post: gamma, vignetting, chromatic aberration
- [x] AA: randomly sampled
- [x] textures: albedo, normal, roughness, metalic
- [x] barrel distortion, fish eye lens
- [x] progressive anti aliasing
- [x] adaptive resolution
- [x] bilinear texture sampling for all supported texture maps
- [x] utilize top-level BVH

## Controls

Two examples of keybindings, one in qwerty with wasd gaming bindings and one in qgmlwy leaving your hands in touch typing position.
Can be rebound to anything you want.

Layout  | Style | Move: up, down, forward, backward, left, right | Look: up, down, left, right | Toggle focus mode | Export frame
--------|-------|------------------------------------------------|-----------------------------|-------------------|---------------
QWERTY  |Gaming | Q, E, W, S, A, D                               | I, K, J, L                  | U                 | O
QGMLWY  |Typing | G, L, M, T, S, N                               | U, E, A, O                  | F                 | B

## Possible things to work on
- rust-gpu rewrite
- wavefront
- portals
- hdr skybox
- sphere skybox only tophalf option
- skybox cubemap
- procedural sky
- denoising
- optimize pow: gamma correct images before upload
- optimize vector loading: use vload3 and allign the buffer for it
- preprocess kernel: optimize branches away, insert constants
- sRGB now, use aces, linear colours
- Next event estimation (NEE)
- Russian roulette (RR)
- Importance sampling of BRDF
- Importance sampling of lights
- Depth of field
- Blue noise
- Multiple importance samplign (MIS)
- Spectral rendering
- motion blur
- path regularization option (biased)
- energy clamp option (biased)

## Sources
1. [Tonemapping; Matt Taylor (2019)](https://64.github.io/tonemapping/)
2. [Tonemapping; Krzysztof Narkowicz (2016)](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/)
3. [Microfacets; Brian Karis (2013)](http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html)
4. [Microfacets; Wanter et al. (2007)](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf)
5. [Microfacets; Joe Schutte (2018)](https://schuttejoe.github.io/post/ggximportancesamplingpart1/)
6. [Microfacets; Jacco Bikker (2016)](https://www.cs.uu.nl/docs/vakken/magr/portfolio/INFOMAGR/lecture13.pdf)
<!---
[Microfacets; Eric Heitz (2018)]https://jcgt.org/published/0007/04/01/paper.pdf
--->
