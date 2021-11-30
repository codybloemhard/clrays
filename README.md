# clrays

## P1 Checklist: Whitted CPU
- [x] architecture
- [x] cam: pos, dir, fov, aspect
- [x] cam: controls
- [x] primitives: planes, spheres
- [x] material
- [x] scene
- [x] blinn shading
- [x] reflection
- [ ] refraction
- [ ] absorption
- [x] multithreading: 4x+
- [x] post: gamma, vignetting, chromatic aberration
- [x] AA: randomly sampled
- [x] textures: albedo, normal, roughness, metalic
- [ ] mesh

## Extra features
- skycolour, skybox(sphere)
- progressive anti aliasing

## Possible things to work on
- random AA using xor32
- cpu renderer
- movement
- pathtracing
- triangle
- portals
- refraction
- hdr skybox
- sphere skybox only tophalf option
- skybox cubemap
- procedural sky
- bi/trilinear texture filtering
- use Wang hash + xor32 on gpu
- accelerating structure
- models
- export
- denoising
- optimize pow: gamma correct images before upload and gamma correct after AA
- optimize vector loading: use vload3 and allign the buffer for it
- preprocess kernel: optimize branches away, insert constants
- sRGB now, use aces, linear colours

