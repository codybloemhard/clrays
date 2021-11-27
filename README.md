# clrays

## Features
- raytracing:
- spheres, planes, box
- albedo (shadows), specular
- reflection
- AA
- textures uv (planes,spheres,box as plane)
- albedo, normal maps,roughness maps,metalic maps
- skycolour, skybox(sphere), sky lighting

## Possible things to work on
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
- accelerating structure
- models
- export
- denoising
- optimize pow: gamma correct images before upload and gamma correct after AA
- optimize vector loading: use vload3 and allign the buffer for it
- preprocess kernel: optimize branches away, insert constants
- sRGB now, use aces, linear colours

## P1 Checklist
- [x] architecture
- [x] cam: pos, dir, fov, aspect
- [ ] cam: controls
- [x] planes n spheres
- [x] material
- [x] scene
- [x] blinn shading
- [x] reflection
- [ ] refraction
- [ ] absorption
- [ ] multithread 4x
- [x] post: gamma, vignetting, chromatic aberration
- [x] aa
- [x] textures
- [ ] mesh
