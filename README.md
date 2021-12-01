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
- [x] skycolour, skybox: sphere
- [x] progressive anti aliasing
- [x] adaptive resolution

## Possible things to work on
- movement on gpu
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
- optimize pow: gamma correct images before upload
- optimize vector loading: use vload3 and allign the buffer for it
- preprocess kernel: optimize branches away, insert constants
- sRGB now, use aces, linear colours

