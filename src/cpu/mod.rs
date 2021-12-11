use crate::scene::{ Scene, Material, Sphere, Plane, Triangle };
use crate::vec3::Vec3;
use crate::state::{ RenderMode, State };

use rand::prelude::*;
use crate::consts::*;
use crate::bvh::{inter_plane, inter_sphere, inter_triangle};

#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
pub fn whitted(
    w: usize, h: usize, threads: usize,
    scene: &Scene, tex_params: &[u32], textures: &[u8],
    screen: &mut Vec<u32>, acc: &mut Vec<Vec3>, state: &mut State, rng: &mut ThreadRng
){
    let reduce = match state.render_mode{
        RenderMode::Reduced => state.reduced_rate,
        _ => 1,
    };

    let rw = w / reduce;
    let rh = h / reduce;
    let pixels = usize::min(rw, rh);
    let radius = pixels as f32 * 0.5;

    if reduce != 1 || state.aa_count == 0 {
        acc.iter_mut().take(rw * rh).for_each(|v| *v = Vec3::ZERO);
    }

    if reduce == 1 && state.aa_count == state.aa{
        state.last_frame = RenderMode::None;
        return;
    } else if reduce == 1{
        state.aa_count += 1;
        state.last_frame = RenderMode::Full;
    } else {
        state.aa_count = 0;
        state.last_frame = RenderMode::Reduced;
    }

    let (aa_count, max_depth) = match state.render_mode{
        RenderMode::Reduced => (1, 1),
        _ => (state.aa_count, state.settings.max_render_depth),
    };

    let threads = threads.max(1);
    let target_strip_h = (rh / threads) + 1;
    let target_strip_l = target_strip_h * rw;
    let strips: Vec<&mut[Vec3]> = acc.chunks_mut(target_strip_l).take(threads).collect();
    let seeds = (0..threads).map(|_| rng.gen::<u32>()).collect::<Vec<_>>();

    crossbeam_utils::thread::scope(|s|{
        let mut handlers = Vec::new();
        for (t, strip) in strips.into_iter().enumerate(){
            let strip_h = strip.len() / rw;
            let offset = t * target_strip_h;
            let seed = seeds[t];

            let pos = scene.cam.pos;
            let cd = scene.cam.dir.normalized_fast();
            let aspect = rw as f32 / rh as f32;
            let uv_dist = (aspect / 2.0) / (scene.cam.fov / 2.0 * 0.01745329).tan();

            // Camera direction in spherical coordinates, phi (x dir) and theta (z dir)
            let phi_mid = f32::atan2(cd.x, -cd.z);
            let theta_mid = cd.y.asin();
            let angle = scene.cam.angle_radius;
            let dist_coef = scene.cam.distortion_coefficient;
            let is_wide = angle > 0.0;

            let handler = s.spawn(move |_|{
                let mut seed = seed;
                for xx in 0..rw{
                for yy in 0..strip_h{
                    let x = xx;
                    let y = yy + offset;
                    let aa_u = u32tf01(xor32(&mut seed));
                    let aa_v = u32tf01(xor32(&mut seed));

                    let dir = initial_ray_dir(pos, cd, x as f32, y as f32, rw as f32, rh as f32, aa_u, aa_v,
                                              aspect, uv_dist, angle, radius, theta_mid, phi_mid, dist_coef, is_wide);
                    let col = if dir.eq(&Vec3::ZERO){
                        Vec3::BLACK
                    } else {
                        let ray = Ray { pos, dir };
                        let contexts = Contexts::new();
                        let mut col = whitted_trace(ray, scene, tex_params, textures, max_depth, contexts);
                        col.pow_scalar(1.0 / GAMMA);
                        col
                    };
                    strip[xx + yy * rw].add(col);
                }
                }
            });
            handlers.push(handler);
        }
        handlers.into_iter().for_each(|h| h.join().expect("Could not join whitted cpu thread (tracing phase)!"));
    }).expect("Could not create crossbeam threadscope (tracing phase)!");

    let ash = scene.cam.chromatic_aberration_shift;
    let ast = scene.cam.chromatic_aberration_strength;
    let vst = scene.cam.vignette_strength;

    let target_strip_h = (h / threads) + 1;
    let target_strip_l = target_strip_h * w;
    let strips: Vec<&mut[u32]> = screen.chunks_mut(target_strip_l).collect();

    let acc: &[Vec3] = acc.as_ref();
    crossbeam_utils::thread::scope(|s|{
        let mut handlers = Vec::new();
        for (t, strip) in strips.into_iter().enumerate(){
            let strip_h = strip.len() / w;
            let offset = t * target_strip_h;
            let handler = s.spawn(move |_|{
                for xx in 0..w{
                for yy in 0..strip_h{
                    let x = xx;
                    let y = yy + offset;
                    let mut uv = Vec3::new(x as f32 / w as f32, y as f32 / h as f32, 0.0);
                    let mut col = acc[(x / reduce).min(rw - 1) + (y / reduce).min(rh - 1) * rw];
                    if ash > 0 && ast > EPSILON{
                        let r = acc[((x.max(ash) - ash) / reduce).min(rw - 1) + ((y.max(ash) - ash) / reduce).min(rh - 1) * rw].x;
                        let b = acc[((x.min(w - ash - 1) + ash) / reduce).min(rw - 1) + ((y.min(h - ash - 1) + ash) / reduce).min(rh - 1) * rw].z;
                        let abr_str = (uv.x * uv.y * 8.0).powf(ast).min(1.0).max(0.0);
                        col.mix(Vec3::new(r, col.y, b), 1.0 - abr_str);
                    }
                    uv.x *= 1.0 - uv.x;
                    uv.y *= 1.0 - uv.y;
                    let vignette = (uv.x * uv.y * 32.0).powf(vst).min(1.0).max(0.0);
                    col.div_scalar_fast(aa_count as f32);
                    col.scale(vignette);
                    col.clamp(0.0, 1.0);
                    col.scale(255.0);
                    let int = ((col.x as u32) << 16) + ((col.y as u32) << 8) + col.z as u32;
                    strip[xx + yy * w] = int;
                }
                }
            });
            handlers.push(handler);
        }
        handlers.into_iter().for_each(|h| h.join().expect("Could not join whitted cpu thread! (post phase)"));
    }).expect("Could not create crossbeam threadscope! (post phase)");
}

// RANDOM ------------------------------------------------------------

// credit: George Marsaglia
#[inline]
fn xor32(seed: &mut u32) -> u32{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

#[inline]
fn u32tf01(int: u32) -> f32{
   int as f32 * 2.3283064e-10
}


// WHITTED ------------------------------------------------------------

// trace light ray through scene
fn whitted_trace(ray: Ray, scene: &Scene, tps: &[u32], ts: &[u8], depth: u8, contexts: Contexts) -> Vec3{
    let mut hit = RayHit::NULL;
    if scene.use_bvh {
        let (bounding_boxes_intersections, primitive_intersections, depth) = scene.bvh.intersect(ray, scene, &mut hit);
        if scene.show_bvh {
            let val = depth;
            let lim1 = 10.0;
            let lim2 = 30.0;
            if bounding_boxes_intersections < lim1 as usize {
                return Vec3 {
                    x: val as f32 / lim1,
                    y: val as f32 / lim1,
                    z: val as f32 / lim1,
                };
            } else {
                // zero is blue
                // 100 is red
                return Vec3 {
                    x: val as f32 / lim2,
                    y: 0.0,
                    z: 1.0 - val as f32 / lim2,
                };
            }
        }
    } else {
        hit = inter_scene(ray, scene);
    };

    // println!("intersections: {:?}", scene.bvh.intersect(ray, scene, &mut hit));


    if depth == 0 || hit.is_null() {
        return get_sky_col(ray.dir, scene, tps, ts);
        // return Vec3{
        //     x: 1.0,
        //     y: 1.0,
        //     z: 1.0,
        // }
    }

    let absorption_context;
    let refraction_context;
    if let Some(context) = contexts.current() {
        absorption_context = context.absorption;
        refraction_context = context.refraction;
    } else {
        absorption_context = Vec3::BLACK;
        refraction_context = 1.0;
    }

    let is_inbound = ray.dir.dot(hit.nor) < 0.0;
    let mat = hit.mat.unwrap();
    let refraction = mat.refraction;
    let absorption = mat.absorption;
    let normal = if is_inbound { hit.nor } else { hit.nor.neged() };
    let mut reflectivity = mat.reflectivity;
    let mut transparency = mat.transparency;

    // texture
    let mut texcol = Vec3::ONE;
    let uv = if
        mat.texture > 0 ||
        mat.normal_map > 0 ||
        mat.roughness_map > 0 ||
        mat.metalic_map > 0 ||
        mat.is_checkerboard
    {
        let uvtype = hit.uvtype;
        let uv = if uvtype == UV_SPHERE{
            sphere_uv(hit.nor)
        } else {
            plane_uv(hit.pos, hit.nor)
        };
        (uv.0 * mat.tex_scale, uv.1 * mat.tex_scale)
    } else {
        (0.0, 0.0)
    };

    // checkerboard custom texture
    if mat.is_checkerboard{
        texcol = if ((uv.0.floor()*3.0) as i32 + (uv.1.floor()*3.0) as i32) % 2 == 0 { Vec3::BLACK } else { Vec3::WHITE }
    } else if mat.texture > 0{
        texcol = get_tex_col(mat.texture - 1, uv, tps, ts);
    }

    // normalmap
    if mat.normal_map > 0{
        let mut rawnor = get_tex_val(mat.normal_map - 1, uv, tps, ts);
        let mut t = Vec3::crossed(hit.nor, Vec3::UP);
        if t.len() < EPSILON{
            t = Vec3::crossed(hit.nor, Vec3::FORWARD);
        }
        t.normalize_fast();
        let b = Vec3::normalized_fast(Vec3::crossed(hit.nor, t));
        rawnor = rawnor.scaled(2.0).added_scalar(-1.0);
        rawnor.normalize_fast();
        let mut newnor = Vec3::ZERO;
        let mut row = Vec3::new(t.x, b.x, hit.nor.x);
        newnor.x = Vec3::dot(row, rawnor);
        row = Vec3::new(t.y, b.y, hit.nor.y);
        newnor.y = Vec3::dot(row, rawnor);
        row = Vec3::new(t.z, b.z, hit.nor.z);
        newnor.z = Vec3::dot(row, rawnor);
        hit.nor = newnor.normalized_fast();
    }

    // roughnessmap
    let mut roughness = mat.roughness;
    if mat.roughness_map > 0{
        let value = get_tex_scalar(mat.roughness_map - 1, uv, tps, ts);
        roughness *= value;
    }

    // metalicmap
    if mat.metalic_map > 0 {
        let value = get_tex_scalar(mat.metalic_map - 1, uv, tps, ts);
        reflectivity *= value;
    }

    // diffuse, specular
    let (mut diff, spec) = blinn(&hit, mat, roughness, scene, ray.dir);
    diff.mul(texcol);

    // dielectric: transparency / refraction and reflection
    let mut transparency_dir = ray.dir;
    if mat.is_dielectric  {
        let ni;
        let nt;
        if is_inbound {
            ni = refraction_context;
            nt = refraction;
        } else {
            ni = refraction;
            nt = refraction_context;
        }
        let (new_ref, new_trans_dir) = resolve_dielectric(ni, nt, ray.dir, normal);
        reflectivity = new_ref;
        transparency_dir = new_trans_dir;
        transparency = 1.0 - reflectivity;
    }

    // transparency / refraction
    let tran = if transparency > EPSILON {
        let ray_next = Ray{ pos: hit.pos.subed(normal.scaled(EPSILON)), dir: transparency_dir };
        let contexts_next = if is_inbound {
            contexts.pushed(Context { absorption, refraction })
        } else {
            contexts.popped()
        };
        whitted_trace(ray_next, scene, tps, ts, depth - 1, contexts_next).scaled(transparency)
    } else { Vec3::BLACK };

    // reflection
    let refl = if reflectivity > EPSILON {
        let ray_next = Ray{ pos: hit.pos.added(normal.scaled(EPSILON)), dir: ray.dir.reflected(normal) };
        whitted_trace(ray_next, scene, tps, ts, depth - 1, contexts).scaled(reflectivity)
    } else { Vec3::BLACK };

    let color = (diff).scaled(1.0 - reflectivity - transparency)
        .added(spec)
        .added(tran)
        .added(refl);

    if is_inbound {
        absorp(&color, &absorption_context, hit.t)
    } else {
        absorp(&color, &absorption, hit.t)
    }
}

// LENS ------------------------------------------------------------

#[inline]
#[allow(clippy::too_many_arguments)]
fn initial_ray_dir(pos: Vec3, cd: Vec3, x: f32, y: f32, rw: f32, rh: f32, aa_u: f32, aa_v: f32,
                   aspect: f32, uv_dist: f32, angle: f32, radius: f32, theta_mid: f32, phi_mid: f32, dist_coef: f32,
                   is_wide: bool) -> Vec3
{
    let u = (x as f32 + aa_u) / rw as f32;
    let v = (y as f32 + aa_v) / rh as f32;
    if !is_wide {
        // normal
        let hor = cd.crossed(Vec3::UP).normalized_fast();
        let ver = hor.crossed(cd).normalized_fast();

        let mut uv = Vec3::new(u, v, 0.0);
        uv.add_scalar(-0.5);
        uv.mul(Vec3::new(aspect, -1.0, 0.0));
        let mut to = pos.added(cd.scaled(uv_dist));
        to.add(hor.scaled(uv.x));
        to.add(ver.scaled(uv.y));
        to.subed(pos).normalized_fast()
    } else {
        // wide-angle
        let dx = (u - 0.5) * aspect;
        let dy = v - 0.5;

        // barrel distortion
        let rho = f32::atan2(dy, dx);
        let r = (dx * dx + dy * dy).sqrt();
        let r = r * 2.0; // ensures r covers range [-1.0, 1.0]
        let r = r.powf(dist_coef);
        let r = r * 0.5; // scales back to range [-0.5, 0.5]
        let dx = r * rho.cos();
        let dy = r * rho.sin();

        // circular screen
        let off_x = x as f32 - rw as f32 * 0.5;
        let off_y = y as f32 - rh as f32 * 0.5;
        let off_len = (off_x * off_x + off_y * off_y).sqrt();
        if off_len > radius {
            Vec3::ZERO
        } else {
            assert!(r <= 1.0);
            assert!(r.powf(dist_coef) <= 1.0);
            let phi = phi_mid + dx * angle * 2.0;
            let theta = theta_mid - dy * angle * 2.0;
            Vec3{
                x:  theta.cos() * phi.sin(),
                y:  theta.sin(),
                z: -theta.cos() * phi.cos()
            }.normalized_fast()
        }
    }
}

// REFRACTION ------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct Contexts {
    pub stack: [Context; 3],
    pub index: usize
}

impl Contexts {
    pub fn new() -> Self{
        Self { stack: [Context::new(), Context::new(), Context::new()], index: 0 }
    }
    pub fn current(&self) -> Option<Context>{
        if self.index <= 2 {
            Some(self.stack[self.index])
        } else {
            None
        }
    }
    pub fn pushed(mut self, context: Context) -> Self{
        self.index = (self.index+1).min(2);
        self.stack[self.index] = context;
        self
    }
    pub fn popped(mut self) -> Self{
        self.index = (self.index-1).max(0);
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Context {
    pub absorption: Vec3,
    pub refraction: f32,
}

impl Context {
    pub fn new() -> Self{
        Self { absorption: Vec3::BLACK, refraction: 1.0 }
    }
}

#[inline]
fn absorp(color: &Vec3, absorption: &Vec3, d: f32) -> Vec3 {
    if absorption.eq(&Vec3::BLACK) {
        *color
    } else {
        Vec3 {
            x: color.x * ((1.0-absorption.x).ln() * d).exp(),
            y: color.y * ((1.0-absorption.y).ln() * d).exp(),
            z: color.z * ((1.0-absorption.z).ln() * d).exp(),
        }
    }
}

// reflectivity and (refracted) ray direction
#[inline]
fn resolve_dielectric(ni: f32, nt: f32, dir: Vec3, normal: Vec3) -> (f32, Vec3){
    // same medium
    if ni == nt { return (0.0, dir) }

    let n = ni / nt;
    let cos_oi = normal.dot(dir.neged());
    let cos_oi2 = cos_oi*cos_oi;
    let sin_oi2 = 1.0 - cos_oi2;
    let sin_ot2 = n*n*sin_oi2;

    // full internal reflection
    if 1.0 - sin_ot2 < 0.0 { return (1.0, Vec3::ZERO); }

    let cos_ot2 = 1.0 - sin_ot2;
    let cos_ot = cos_ot2.sqrt();
    let dir_t = normal.scaled(-cos_oi);
    let dir_p = dir.subed(dir_t);
    let refr_t = normal.scaled(-(1.0-sin_ot2).sqrt());
    let refr_p = dir_p.scaled(n);
    let dir_refr = refr_p.added(refr_t);

    // fresnell
    let _a1 = (ni * cos_oi - nt * cos_ot) /
              (ni * cos_oi + nt * cos_ot);
    let _a2 = (ni * cos_ot - nt * cos_oi) /
              (ni * cos_ot + nt * cos_oi);
    let s_polarized = _a1 * _a1;
    let p_polarized = _a2 * _a2;
    (0.5 * (s_polarized + p_polarized), dir_refr)
}

// SHADING ------------------------------------------------------------

// get diffuse light incl colour of hit with all lights
fn blinn(hit: &RayHit, mat: &Material, roughness: f32, scene: &Scene, viewdir: Vec3) -> (Vec3, Vec3){
    let mut col = Vec3::ONE.scaled(AMBIENT);
    let mut spec = Vec3::ZERO;
    for light in &scene.lights{
        let res = blinn_single(roughness, light.pos, light.intensity, viewdir, hit, scene);
        col.add(light.col.scaled(res.0));
        spec.add(light.col.scaled(res.1));
    }
    (col.muled(mat.col), spec.scaled(1.0 - roughness))
}

// get diffuse light strength for hit for a light
fn blinn_single(roughness: f32, lpos: Vec3, lpow: f32, viewdir: Vec3, hit: &RayHit, scene: &Scene) -> (f32, f32){
    let mut to_l = Vec3::subed(lpos, hit.pos);
    let dist = Vec3::len(to_l);
    to_l.scale(1.0 / (dist + EPSILON));
    // diffuse
    let mut angle = Vec3::dot(hit.nor, to_l);
    if angle < EPSILON{
        return (0.0, 0.0);
    }
    angle = angle.max(0.0);
    let power = lpow / (PI * 4.0 * dist * dist);
    if power < EPSILON{
        return (0.0, 0.0);
    }
    // exposed to light or not
    let lray = Ray { pos: hit.pos.added(hit.nor.scaled(EPSILON)), dir: to_l };
    if is_occluded(lray, scene, dist){
        return (0.0, 0.0);
    }
    // specular
    let halfdir = Vec3::normalized_fast(to_l.subed(viewdir));
    let specangle = Vec3::dot(halfdir, hit.nor).max(0.0);
    let spec = specangle.powf(16.0 / roughness);
    (angle * power, spec * power)
}

// UV's ------------------------------------------------------------

// plane uv
#[inline]
fn plane_uv(pos: Vec3, nor: Vec3) -> (f32, f32){
    let u = Vec3::new(nor.y, nor.z, -nor.x);
    let v = Vec3::crossed(u, nor).normalized_fast();
    (Vec3::dot(pos, u), Vec3::dot(pos, v))
}

// sphere uv
#[inline]
fn sphere_uv(nor: Vec3) -> (f32, f32){
    let u = 0.5 + (f32::atan2(-nor.z, -nor.x) / (2.0 * PI));
    let v = 0.5 - (f32::asin(-nor.y) / PI);
    (u, v)
}

// sphere skybox uv(just sphere uv with inverted normal)
#[inline]
fn sky_sphere_uv(nor: Vec3) -> (f32, f32){
    let u = 0.5 + (f32::atan2(nor.z, nor.x) / (2.0 * PI));
    let v = 0.5 - (f32::asin(nor.y) / PI);
    (u, v)
}

// INTERSECTING ------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Ray{
    pub pos: Vec3,
    pub dir: Vec3,
}
impl Ray {
    #[inline]
    pub fn inverted(mut self) -> Self{
        self.dir = Vec3 {
            x: if self.dir.x.abs() > EPSILON { 1.0 / self.dir.x } else { self.dir.x.signum() * f32::INFINITY },
            y: if self.dir.y.abs() > EPSILON { 1.0 / self.dir.y } else { self.dir.y.signum() * f32::INFINITY },
            z: if self.dir.z.abs() > EPSILON { 1.0 / self.dir.z } else { self.dir.z.signum() * f32::INFINITY },
        };
        self
    }

    #[inline]
    pub fn direction_negations(self) -> [usize; 3]{
        [
            (self.dir.x < 0.0) as usize,
            (self.dir.y < 0.0) as usize,
            (self.dir.z < 0.0) as usize,
        ]
    }
}

#[derive(Clone)]
pub struct RayHit<'a>{
    pub pos: Vec3,
    pub nor: Vec3,
    pub t: f32,
    pub mat: Option<&'a Material>,
    pub uvtype: u8,
    pub sphere: Option<&'a Sphere>,
}

impl RayHit<'_>{
    pub const NULL: Self = RayHit{
        pos: Vec3::ZERO,
        nor: Vec3::ZERO,
        t: MAX_RENDER_DIST,
        mat: None,
        uvtype: 255,
        sphere: None,
    };

    #[inline]
    pub fn is_null(&self) -> bool{
        self.uvtype == 255
    }
}

#[inline]
fn dist_sphere(ray: Ray, sphere: &Sphere) -> f32{
    let l = Vec3::subed(sphere.pos, ray.pos);
    let tca = Vec3::dot(ray.dir, l);
    let d = tca*tca - Vec3::dot(l, l) + sphere.rad*sphere.rad;
    if d < 0.0 { return MAX_RENDER_DIST; }
    let dsqrt = d.sqrt();
    let mut t = tca - dsqrt;
    if t < 0.0 {
        t = tca + dsqrt;
        if t < 0.0 { return MAX_RENDER_DIST; }
    }
    t
}

#[inline]
fn dist_plane(ray: Ray, plane: &Plane) -> f32{
    let divisor = Vec3::dot(ray.dir, plane.nor);
    if divisor.abs() < EPSILON { return MAX_RENDER_DIST; }
    let planevec = Vec3::subed(plane.pos, ray.pos);
    let t = Vec3::dot(planevec, plane.nor) / divisor;
    if t < EPSILON { return MAX_RENDER_DIST; }
    t
}

#[inline]
#[allow(clippy::many_single_char_names)]
fn dist_triangle(ray: Ray, tri: &Triangle) -> f32{
    let edge1 = Vec3::subed(tri.b, tri.a);
    let edge2 = Vec3::subed(tri.c, tri.a);
    let h = Vec3::crossed(ray.dir, edge2);
    let a = Vec3::dot(edge1, h);
    if a > -EPSILON && a < EPSILON { return MAX_RENDER_DIST; } // ray parallel to tri
    let f = 1.0 / a;
    let s = Vec3::subed(ray.pos, tri.a);
    let u = f * Vec3::dot(s, h);
    if !(0.0..=1.0).contains(&u) { return MAX_RENDER_DIST; }
    let q = Vec3::crossed(s, edge1);
    let v = f * Vec3::dot(ray.dir, q);
    if v < 0.0 || u + v > 1.0 { return MAX_RENDER_DIST; }
    let t = f * Vec3::dot(edge2, q);
    if t <= EPSILON { return MAX_RENDER_DIST; }
    t
}

// intersect whole scene, find closest hit
#[inline]
fn inter_scene(ray: Ray, scene: &Scene) -> RayHit{
    let mut closest = RayHit::NULL;
    for plane in &scene.planes { inter_plane(ray, plane, &mut closest); }
    for sphere in &scene.spheres { inter_sphere(ray, sphere, &mut closest); }
    for tri in &scene.triangles { inter_triangle(ray, tri, &mut closest); }
    closest
}

// intersect whole scene, stop at first intersection at all
#[inline]
fn is_occluded(ray: Ray, scene: &Scene, ldist: f32) -> bool{
    for plane in &scene.planes {
        if dist_plane(ray, plane) < ldist { return true; }
    }
    for sphere in &scene.spheres {
        if dist_sphere(ray, sphere) < ldist { return true; }
    }
    for tri in &scene.triangles {
        if dist_triangle(ray, tri) < ldist { return true; }
    }
    false
}

// TEXTURES ------------------------------------------------------------

// first byte of texture
#[inline]
fn tx_get_start(tex: u32, tps: &[u32]) -> usize{
    tps[tex as usize * 3] as usize
}

#[inline]
fn tx_get_width(tex: u32, tps: &[u32]) -> u32{
    tps[tex as usize * 3 + 1]
}

#[inline]
fn tx_get_height(tex: u32, tps: &[u32]) -> u32{
    tps[tex as usize * 3 + 2]
}

// get sample
#[inline]
fn tx_get_sample(tex: u32, tps: &[u32], ts: &[u8], x: u32, y: u32, w: u32) -> Vec3{
    let offset = tx_get_start(tex, tps) + ((y * w + x) * 3) as usize;
    let col = Vec3::new(ts[offset    ] as f32,
                        ts[offset + 1] as f32,
                        ts[offset + 2] as f32);
    col.dived_scalar_fast(255.0)
}

// get value 0..1 from scalar map
#[inline]
fn tx_get_scalar(tex: u32, tps: &[u32], ts: &[u8], x: u32, y: u32, w: u32) -> f32{
    let offset = tx_get_start(tex, tps) + ((y * w + x) as usize);
    if offset > ts.len() { return 1.0; }
    let scalar = ts[offset] as f32;
    scalar / 255.0
}

// shared logic
#[inline]
#[allow(clippy::many_single_char_names)]
fn uv_to_xy(uv: (f32, f32), tex: u32, tps: &[u32]) -> (u32, f32, f32){
    let mut u = uv.0.fract();
    let mut v = uv.1.fract();
    if u < 0.0 { u += 1.0; }
    if v < 0.0 { v += 1.0; }
    let w = tx_get_width(tex, tps) as f32;
    let x = w * u;
    let y = tx_get_height(tex, tps) as f32 * v;
    (w as u32, x, y)
}

// get sky colour
#[inline]
fn get_sky_col(nor: Vec3, scene: &Scene, tps: &[u32], ts: &[u8]) -> Vec3{
    if scene.sky_box == 0{
        return scene.sky_col;
    }
    let uv = sky_sphere_uv(nor);
    get_tex_col(scene.sky_box - 1, uv, tps, ts)
}

// get value to range 0..1 (no gamma)
#[inline]
#[allow(clippy::many_single_char_names)]
fn get_tex_val(tex: u32, uv: (f32, f32), tps: &[u32], ts: &[u8]) -> Vec3{
    let (w, x, y) = uv_to_xy(uv, tex, tps);
    let (x1, y1, x2, y2) = (x.floor(), y.floor(), x.floor() + 1.0, y.floor() + 1.0);
    let fq11 = tx_get_sample(tex, tps, ts, x1 as u32, y1 as u32, w);
    let fq12 = tx_get_sample(tex, tps, ts, x1 as u32, y2 as u32, w);
    let fq21 = tx_get_sample(tex, tps, ts, x2 as u32, y1 as u32, w);
    let fq22 = tx_get_sample(tex, tps, ts, x2 as u32, y2 as u32, w);
    let a = x - x1;
    let b = y - y1;
    fq11.scaled((1.0 - a) * (1.0 - b))
        .added(fq21.scaled(a * (1.0 - b)))
        .added(fq12.scaled(b * (1.0 - a)))
        .added(fq22.scaled(a * b))
}

// get colour from texture and uv
#[inline]
fn get_tex_col(tex: u32, uv: (f32, f32), tps: &[u32], ts: &[u8]) -> Vec3{
    get_tex_val(tex, uv, tps, ts).powed_scalar(GAMMA)
}

// get value 0..1 from scalar map
#[inline]
#[allow(clippy::many_single_char_names)]
fn get_tex_scalar(tex: u32, uv: (f32, f32), tps: &[u32], ts: &[u8]) -> f32{
    let (w, x, y) = uv_to_xy(uv, tex, tps);
    let (x1, y1, x2, y2) = (x.floor(), y.floor(), x.floor() + 1.0, y.floor() + 1.0);
    let fq11 = tx_get_scalar(tex, tps, ts, x1 as u32, y1 as u32, w);
    let fq12 = tx_get_scalar(tex, tps, ts, x1 as u32, y2 as u32, w);
    let fq21 = tx_get_scalar(tex, tps, ts, x2 as u32, y1 as u32, w);
    let fq22 = tx_get_scalar(tex, tps, ts, x2 as u32, y2 as u32, w);
    let a = x - x1;
    let b = y - y1;
    fq11 * (1.0 -a) * (1.0 - b)
        + fq21 * a * (1.0 - b)
        + fq12 * b * (1.0 - a)
        + fq22 * a * b
}

// TESTS ------------------------------------------------------------

#[cfg(test)]
mod test {
    use crate::vec3::Vec3;
    use crate::cpu::{resolve_dielectric, EPSILON, FRAC_4_PI, Ray};
    use crate::consts::{ FRAC_2_PI, PI };

    fn assert_small(a:f32,b:f32) {
        if (a-b).abs() > EPSILON { panic!("{} != {}", a,b); }
    }

    fn assert_small_vec(a: Vec3, b: Vec3) {
        if (a.x-b.x).abs() > EPSILON { panic!("{:?} != {:?}", a,b); }
        if (a.y-b.y).abs() > EPSILON { panic!("{:?} != {:?}", a,b); }
        if (a.z-b.z).abs() > EPSILON { panic!("{:?} != {:?}", a,b); }
    }

    #[test]
    fn test_computing_angles() {
        let dir = Vec3::DOWN;
        let normal = Vec3::UP;
        let cos_oi = dir.neged().dot(normal);
        let sin_oi = dir.neged().crossed(normal).len();
        assert_small(cos_oi, 1.0);
        assert_small(sin_oi, 0.0);

        let dir = Vec3 {x: FRAC_4_PI.cos(), y: -FRAC_4_PI.sin(), z: 0.0};
        let normal = Vec3::UP;
        let cos_oi = dir.neged().dot(normal);
        let sin_oi = dir.neged().crossed(normal).len();
        assert_small(cos_oi, FRAC_4_PI.cos());
        assert_small(sin_oi, FRAC_4_PI.sin());

        let sin_oi2 = 1.0 - cos_oi * cos_oi;
        assert_small(sin_oi, sin_oi2.sqrt());
    }
    #[test]
    fn test_computing_angles_opposite_half_refraction_stupid() {
        let dir = Vec3::DOWN;
        let normal = Vec3::UP;
        let n = 1.0 / 2.0;
        let cos_oi = dir.neged().dot(normal);
        assert_small(cos_oi, 1.0);
        let sin_oi2 = 1.0 - cos_oi * cos_oi;
        let sin_oi = sin_oi2.sqrt();
        assert_small(sin_oi, 0.0);
        let sin_ot2 = n*n*sin_oi2;
        let sin_ot = sin_oi2.sqrt();
        assert_small(sin_ot, 0.0);
        let cos_ot2= 1.0 - sin_ot2;
        let cos_ot = cos_ot2.sqrt();
        assert_small(cos_ot, 1.0);

        let dir_t = normal.scaled(-cos_oi);
        let dir_p = dir.clone().subed(dir_t);

        let refr_t = normal.scaled(-(1.0-sin_ot2).sqrt());
        let refr_p = dir_p.scaled(n);

        assert_eq!(dir_p,refr_p);
        assert_eq!(dir_t,refr_t);

        let dir_refr = refr_p.added(refr_t);

        assert_eq!(dir,dir_refr);
    }

    #[test]
    fn test_computing_angles_45_deg_no_refraction_all_steps() {
        let dir = Vec3 {x: FRAC_4_PI.cos(), y: -FRAC_4_PI.sin(), z: 0.0};
        let normal = Vec3::UP;
        let n = 1.0;
        let cos_oi = dir.neged().dot(normal);
        assert_small(cos_oi, FRAC_4_PI.cos());
        let sin_oi2 = 1.0 - cos_oi * cos_oi;
        let sin_oi = sin_oi2.sqrt();
        assert_small(sin_oi, FRAC_4_PI.sin());
        let sin_ot2 = n*n*sin_oi2;
        let sin_ot = sin_oi2.sqrt();
        assert_small(sin_ot, FRAC_4_PI.sin());
        let cos_ot2= 1.0 - sin_ot2;
        let cos_ot = cos_ot2.sqrt();
        assert_small(cos_ot, FRAC_4_PI.cos());

        let dir_t = normal.scaled(-cos_oi);
        let dir_p = dir.clone().subed(dir_t);

        let refr_t = normal.scaled(-(1.0-sin_ot2).sqrt());
        let refr_p = dir_p.scaled(n);

        assert_eq!(dir_p,refr_p);
        assert_eq!(dir_t,refr_t);

        let dir_refr = refr_p.added(refr_t);

        assert_eq!(dir,dir_refr);
    }

    #[test]
    fn test_computing_angles_45_deg_half_refraction_function() {
        let dir = Vec3 {x: FRAC_4_PI.cos(), y: -FRAC_4_PI.sin(), z: 0.0};
        let normal = Vec3::UP;
        let ni = 0.5;
        let nt = 1.0;
        let n = ni/nt;

        let (_, dir_refr) = resolve_dielectric(ni, nt, dir, normal);

        // Check
        let x = n*dir.x;
        let y = (1.0-x*x).sqrt();
        assert_eq!(Vec3 {x: x, y: -y, z: 0.0},dir_refr);
    }

    #[test]
    fn angles() {
        let cd = Vec3::BACKWARD;
        let angle_radius = FRAC_2_PI;

        let phi_mid = f32::atan2(cd.x,-cd.z);
        let theta_mid = cd.y.asin();
        assert_eq!(phi_mid, 0.0);
        assert_eq!(theta_mid, 0.0);

        // left
        let dx = -0.5;
        let dy = 0.0;
        let phi = phi_mid - dx * angle_radius * 2.0;
        let theta = theta_mid + dy * angle_radius * 2.0;
        let phi = if theta.abs() > FRAC_2_PI {
            phi + PI
        } else { phi };

        let dir = Vec3{
            x: theta.cos()*phi.sin(),
            y: theta.sin(),
            z: -theta.cos()*phi.cos()
        }.normalized_fast();

        assert_small_vec(dir, Vec3::LEFT);

        // right
        let dx = 0.5;
        let dy = 0.0;
        let phi = phi_mid - dx * angle_radius * 2.0;
        let theta = theta_mid + dy * angle_radius * 2.0;
        let phi = if theta.abs() > FRAC_2_PI {
            phi + PI
        } else { phi };

        let dir = Vec3{
            x: theta.cos()*phi.sin(),
            y: theta.sin(),
            z: -theta.cos()*phi.cos()
        }.normalized_fast();

        assert_small_vec(dir, Vec3::RIGHT);

        // up
        let dx = 0.0;
        let dy = 0.5;
        let phi = phi_mid - dx * angle_radius * 2.0;
        let theta = theta_mid + dy * angle_radius * 2.0;
        let phi = if theta.abs() > FRAC_2_PI {
            phi + PI
        } else { phi };

        let dir = Vec3{
            x: theta.cos()*phi.sin(),
            y: theta.sin(),
            z: -theta.cos()*phi.cos()
        }.normalized_fast();

        assert_small_vec(dir, Vec3::UP);

        // down
        let dx = 0.0;
        let dy = -0.5;
        let phi = phi_mid - dx * angle_radius * 2.0;
        let theta = theta_mid + dy * angle_radius * 2.0;
        let phi = if theta.abs() > FRAC_2_PI {
            phi + PI
        } else { phi };

        let dir = Vec3{
            x: theta.cos()*phi.sin(),
            y: theta.sin(),
            z: -theta.cos()*phi.cos()
        }.normalized_fast();

        assert_small_vec(dir, Vec3::DOWN);
    }

    #[test]
    fn screen_pos() {
        let rw = 1200.0;
        let rh = 1000.0;
        // middle
        let xx = 600.0;
        let yy = 500.0;
        let dx = xx / rw - 0.5;
        let dy = yy / rh - 0.5;
        assert_small(dx, 0.0);
        assert_small(dy, 0.0);
        // left
        let xx = 000.0;
        let yy = 500.0;
        let dx = xx / rw - 0.5;
        let dy = yy / rh - 0.5;
        assert_small(dx, -0.5);
        assert_small(dy, 0.0);
        // top
        let xx = 600.0;
        let yy = 1000.0;
        let dx = xx / rw - 0.5;
        let dy = yy / rh - 0.5;
        assert_small(dx, 0.0);
        assert_small(dy, 0.5);
    }

    #[test]
    fn circular_screen() {
        let rw = 1200.0;
        let rh = 1000.0;
        let pixels = f32::min(rw, rh);
        let radius = pixels * 0.5;
        // middle
        let xx = 600.0;
        let yy = 500.0;
        let offset_x = xx - rw * 0.5;
        let offset_y = yy - rh * 0.5;
        let offset_len = (offset_x*offset_x + offset_y*offset_y).sqrt();
        let is_black = offset_len > radius;
        assert_eq!(is_black, false);
        // left
        let xx = 000.0;
        let yy = 500.0;
        let offset_x = xx - rw * 0.5;
        let offset_y = yy - rh * 0.5;
        let offset_len = (offset_x*offset_x + offset_y*offset_y).sqrt();
        let is_black = offset_len > radius;
        assert_eq!(is_black, true);
        // top
        let xx = 600.0;
        let yy = 1000.0;
        let offset_x = xx - rw * 0.5;
        let offset_y = yy - rh * 0.5;
        let offset_len = (offset_x*offset_x + offset_y*offset_y).sqrt();
        let is_black = offset_len > radius;
        assert_eq!(is_black, false);
    }

    #[test]
    fn ray_negated_dir() {
        let ray = Ray {
            pos: Vec3::ZERO,
            dir: Vec3::ONE.normalized()
        };
        assert_eq!(ray.direction_negations(), [0,0,0]);

        let ray = Ray {
            pos: Vec3::ZERO,
            dir: Vec3::ONE.neged().normalized()
        };
        assert_eq!(ray.direction_negations(), [1,1,1]);
    }

    #[test]
    fn ray_inv_dir() {
        let ray = Ray {
            pos: Vec3::ZERO,
            dir: Vec3::ONE.normalized()
        };
        let inv_dir = ray.inverted().dir;
        assert!(inv_dir.x > 1.0 && inv_dir.x < 2.0);
        assert!(inv_dir.y > 1.0 && inv_dir.y < 2.0);
        assert!(inv_dir.z > 1.0 && inv_dir.z < 2.0);

        let ray = Ray {
            pos: Vec3::ZERO,
            dir: Vec3::FORWARD
        };
        let inv_dir = ray.inverted().dir;
        assert!(inv_dir.x > 1.0 && inv_dir.x == f32::INFINITY);
        assert!(inv_dir.y > 1.0 && inv_dir.y == f32::INFINITY);
        assert!(inv_dir.z == 1.0);
    }
}
