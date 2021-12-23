# clrays

## P2 Checklist: GPU Pathtracing with BVH
### BVH
- [x] Correct BVH: 10k+ tris
- [x] BVH Ray/Scene intersection
- [x] BVH Ray/Scene occlusion test
- [x] BVH surface area heuristic
- [x] BVH binning
- [ ] Construct a BVH for dynamic scenery using specialized builders for various types of animation. Add a top-
level BVH to combine the resulting sub-BVHs, and adapt your traversal code to handle rigid motion. Provide
a demo to prove that your BVH handles animated scenes. 2pts
- [ ] Construct Sah BVH for 5M tris scene in less than 1 second, 1pt
- [ ] Construct a 4-way BVH by collapsing a 2-way BVH, and traverse this structure. The resulting traversal speed
must be an improvement over 2-way BVH traversal (good for Kajiya, packets are better for Whitted). 1pt
- [x] Render a 1B poly scene in 5 seconds or less, 5pts
- [x] GPU BVH traversal
### GPU Pathtracing
- [ ] Basic pathtracer (area lights, materials, speculars, dielectrics, beer's law)
- [ ] Frame energy
- [x] Accumulator buffer
- [x] Movement
- [ ] Next event estimation (NEE), 1pt
- [ ] Russian roulette (RR), 0.5pts
- [ ] Importance sampling of BRDF, 0.5pts
- [ ] Importance sampling of lights, 1pt
- [ ] Depth of field, 0.5pts
- [ ] Blue noise, 1 pt
- [ ] Multiple importance samplign (MIS), 1.5pts
- [ ] Spectral rendering, 3pts
- [ ] Motion blur (eh), 0.5pts

## P1 Checklist: Whitted CPU
- [x] architecture
- [x] cam: pos, dir, fov, aspect
- [x] cam controls
  - movement: up, down, forward, backward, left, right
  - looking: up, down, left right
- [x] primitives: planes, spheres
- [x] material
- [x] scene
- [x] blinn shading
- [x] reflection
- [x] refraction
- [x] absorption
- [x] multithreading: 4x+
- [x] post: gamma, vignetting, chromatic aberration
- [x] AA: randomly sampled
- [x] textures: albedo, normal, roughness, metalic
- [x] mesh: triangle meshes (.obj)
- [x] barrel distortion, fish eye lens

## Extra features
- [x] skycolour, skybox: sphere
- [x] progressive anti aliasing
  - can be toggled on runtime, between no AA or progressive AA
- [x] adaptive resolution
- [x] bilinear texture sampling for all supported texture maps
- [x] controls: custom keybindings
- [x] export frame

## Controls

Two examples of keybindings, one in qwerty with wasd gaming bindings and one in qgmlwy leaving your hands in touch typing position.
Can be rebound to anything you want.

Layout  | Style | Move: up, down, forward, backward, left, right | Look: up, down, left, right | Toggle focus mode | Export frame
--------|-------|------------------------------------------------|-----------------------------|-------------------|---------------
QWERTY  |Gaming | Q, E, W, S, A, D                               | I, K, J, L                  | U                 | O
QGMLWY  |Typing | G, L, M, T, S, N                               | U, E, A, O                  | F                 | B

## Possible things to work on
- pathtracing
- portals
- hdr skybox
- sphere skybox only tophalf option
- skybox cubemap
- procedural sky
- use Wang hash + xor32 on gpu
- denoising
- optimize pow: gamma correct images before upload
- optimize vector loading: use vload3 and allign the buffer for it
- preprocess kernel: optimize branches away, insert constants
- sRGB now, use aces, linear colours

