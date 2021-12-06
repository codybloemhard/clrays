# clrays

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
- movement on gpu
- pathtracing
- portals
- hdr skybox
- sphere skybox only tophalf option
- skybox cubemap
- procedural sky
- use Wang hash + xor32 on gpu
- accelerating structure
- denoising
- optimize pow: gamma correct images before upload
- optimize vector loading: use vload3 and allign the buffer for it
- preprocess kernel: optimize branches away, insert constants
- sRGB now, use aces, linear colours

