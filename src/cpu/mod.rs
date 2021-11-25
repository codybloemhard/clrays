use crate::scene::{ Scene, Material, Sphere, Plane };
use crate::vec3::Vec3;

const AA: f32 = 1.0;
const MAX_RENDER_DEPTH: u8 = 3;
const GAMMA: f32 = 2.2;
const PI: f32 = std::f32::consts::PI;
const MAX_RENDER_DIST: f32 = 1000000.0;
const EPSILON:f32 = 0.001;

pub fn test(w: usize, h: usize, screen: &mut Vec<u32>){
    for x in 0..w{
    for y in 0..h{
        let mut uv = Vec3::new(x as f32 / w as f32, y as f32 / h as f32, 0.0);
        uv.add_scalar(-0.5);
        uv.mul(Vec3::new(w as f32 / h as f32, -1.0, 0.0));
        let val = (Vec3::ZERO.dist(uv) * 255.0).min(255.0) as u32;
        screen[x + y * w] = (val << 16) + (val << 8) + val;
    }
    }
}

pub fn whitted(w: usize, h: usize, scene: &Scene, screen: &mut Vec<u32>, tex_params: &[u32], textures: &[u8]){
    for x in 0..w{
    for y in 0..h{
        let pos = scene.cam_pos;
        let cd = scene.cam_dir.normalized_fast();
        let hor = cd.crossed(Vec3::UP).normalized_fast();
        let ver = hor.crossed(cd).normalized_fast();
        let mut uv = Vec3::new(x as f32 / (w as f32 * AA), y as f32 / (h as f32 * AA), 0.0);
        uv.add_scalar(-0.5);
        uv.mul(Vec3::new(w as f32 / h as f32, -1.0, 0.0));
        let mut to = pos.added(cd);
        to.add(hor.scaled(uv.x));
        to.add(ver.scaled(uv.y));
        let ray = Ray{ pos, dir: to.subed(pos).normalized_fast() };

        let mut col = whitted_trace(ray, scene, tex_params, textures, MAX_RENDER_DEPTH);
        col.pow_scalar(1.0 / GAMMA);
        if AA == 1.0{
            col.clamp(0.0, 1.0);
        }
        col.div_scalar_fast(AA * AA);
        screen[x + y * w] = (((col.x * 255.0) as u32) << 16) + (((col.y * 255.0) as u32) << 8) + (col.z * 255.0) as u32;
    }
    }
}

fn whitted_trace(ray: Ray, scene: &Scene, tps: &[u32], ts: &[u8], depth: u8) -> Vec3{
    if depth == 0 {
        return sky_col(ray.dir, scene, tps, ts);
    }

    // hit
    let hit = inter_scene(ray, scene);
    if hit.is_null(){
        return sky_col(ray.dir, scene, tps, ts);
    } else {
        return Vec3::ONE;
    }

    // // texture
    // float2 uv;
    // float3 texcol = (float3)(1.0f);
    // if(hit.mat->texture > 0){
    //     uchar uvtype = hit.mat->uvtype;
    //     if(uvtype == uvPLANE)
    //         uv = PlaneUV(hit.pos, hit.nor);
    //     else if(uvtype == uvSPHERE)
    //         uv = SphereUV(hit.nor);
    //     else if(uvtype == uvBOX)
    //         uv = PlaneUV(hit.pos, hit.nor);
    //     uv *= hit.mat->texscale;
    //     texcol = GetTexCol(hit.mat->texture - 1, uv, scene);
    // }
    //
    // // normalmap
    // if(hit.mat->normalmap > 0){
    //     float3 rawnor = GetTexVal(hit.mat->normalmap - 1, uv, scene);
    //     float3 t = fast_normalize(cross(hit.nor, (float3)(0.0f,1.0f,0.0f)));
    //     if(fast_length(t) < EPSILON)
    //         t = fast_normalize(cross(hit.nor, (float3)(0.0f,0.0f,1.0f)));
    //     t = fast_normalize(t);
    //     float3 b = fast_normalize(cross(hit.nor, t));
    //     rawnor = rawnor * 2 - 1;
    //     rawnor = fast_normalize(rawnor);
    //     float3 newnor;
    //     float3 row = (float3)(t.x, b.x, hit.nor.x);
    //     newnor.x = dot(row, rawnor);
    //     row = (float3)(t.y, b.y, hit.nor.y);
    //     newnor.y = dot(row, rawnor);
    //     row = (float3)(t.z, b.z, hit.nor.z);
    //     newnor.z = dot(row, rawnor);
    //     hit.nor = fast_normalize(newnor);
    // }
    //
    // //roughnessmap
    // if(hit.mat->roughnessmap > 0){
    //     float value = GetTexScalar(hit.mat->roughnessmap - 1, uv, scene);
    //     hit.mat->roughness = value * hit.mat->roughness;
    // }
    //
    // //metalicmap
    // if(hit.mat->metalicmap > 0){
    //     float value = GetTexScalar(hit.mat->metalicmap - 1, uv, scene);
    //     hit.mat->reflectivity *= value;
    // }
    //
    // //diffuse, specular
    // float3 diff, spec;
    // Blinn(&hit, scene, ray->dir, &diff, &spec);
    // diff *= texcol;
    //
    // //reflection
    // float3 newdir = fast_normalize(reflect(ray->dir, hit.nor));
    // struct Ray nray;
    // nray.pos = hit.pos + newdir * EPSILON;
    // nray.dir = newdir;
    // struct RayHit nhit = InterScene(&nray, scene);
    //
    // //Does not get corrupted to version inside recursive call if not pointer
    // float refl_mul = hit.mat->reflectivity;
    // float3 refl = RayTrace(&nray, scene, depth - 1);
    // return (diff * (1.0f - refl_mul)) + (refl * refl_mul) + spec;
}

// sphere skybox uv(just sphere uv with inverted normal)
fn sky_sphere_uv(nor: Vec3) -> (f32, f32){
    let u = 0.5 + (f32::atan2(nor.z, nor.x) / (2.0 * PI));
    let v = 0.5 - (f32::asin(nor.y) / PI);
    (u, v)
}

// get sky colour
fn sky_col(nor: Vec3, scene: &Scene, tps: &[u32], ts: &[u8]) -> Vec3{
    if scene.sky_box == 0{
        return scene.sky_col;
    }
    let uv = sky_sphere_uv(nor);
    get_tex_col(scene.sky_box - 1, uv, tps, ts)
}

// INTERSECTING ------------------------------------------------------------

const UV_PLANE: u8 = 0;
const UV_SPHERE: u8 = 1;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
struct Ray{
    pub pos: Vec3,
    pub dir: Vec3,
}

#[derive(Clone)]
struct RayHit<'a>{
    pub pos: Vec3,
    pub nor: Vec3,
    pub t: f32,
    pub mat: Option<&'a Material>,
    pub uvt: u8,
}

impl RayHit<'_>{
    pub const NULL: Self = RayHit{ pos: Vec3::ZERO, nor: Vec3::ZERO, t: MAX_RENDER_DIST, mat: None, uvt: 255 };

    pub fn is_null(&self) -> bool{
        self.uvt == 255
    }
}

// ray-sphere intersection
#[inline]
fn inter_sphere<'a>(ray: Ray, sphere: &'a Sphere, closest: &mut RayHit<'a>){
    let l = Vec3::subed(sphere.pos, ray.pos);
    let tca = Vec3::dot(ray.dir, l);
    let d = tca*tca - Vec3::dot(l, l) + sphere.rad*sphere.rad;
    if d < 0.0 { return; }
    let dsqrt = d.sqrt();
    let mut t = tca - dsqrt;
    if t < 0.0 {
        t = tca + dsqrt;
        if t < 0.0 { return; }
    }
    if t > closest.t { return; }
    closest.t = t;
    closest.pos = ray.pos.added(ray.dir.scaled(t));
    closest.nor = Vec3::subed(closest.pos, sphere.pos).scaled(1.0 / sphere.rad);
    closest.mat = Some(&sphere.mat);
    closest.uvt = UV_SPHERE;
}

// ray-plane intersection
#[inline]
fn inter_plane<'a>(ray: Ray, plane: &'a Plane, closest: &mut RayHit<'a>){
    let divisor = Vec3::dot(ray.dir, plane.nor);
    if divisor.abs() < EPSILON { return; }
    let planevec = Vec3::subed(plane.pos, ray.pos);
    let t = Vec3::dot(planevec, plane.nor) / divisor;
    if t < EPSILON { return; }
    if t > closest.t { return; }
    closest.t = t;
    closest.pos = ray.pos.added(ray.dir.scaled(t));
    closest.nor = plane.nor;
    closest.mat = Some(&plane.mat);
    closest.uvt = UV_PLANE;
}

// intersect whole scene
fn inter_scene(ray: Ray, scene: &Scene) -> RayHit{
    let mut closest = RayHit::NULL;
    for sphere in &scene.spheres { inter_sphere(ray, sphere, &mut closest); }
    for plane in &scene.planes { inter_plane(ray, plane, &mut closest); }
    closest
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
fn tx_get_sample(tex: u32, tps: &[u32], ts: &[u8], x: u32, y: u32, w: u32) -> Vec3{
    let offset = tx_get_start(tex, tps) + ((y * w + x) * 3) as usize;
    let col = Vec3::new(ts[offset    ] as f32,
                        ts[offset + 1] as f32,
                        ts[offset + 2] as f32);
    col.dived_scalar_fast(255.0)
}

// shared logic
#[inline]
#[allow(clippy::many_single_char_names)]
fn uv_to_xy(uv: (f32, f32), tex: u32, tps: &[u32]) -> (u32, u32, u32){
    let mut u = uv.0.fract();
    let mut v = uv.1.fract();
    if u < 0.0 { u += 1.0; }
    if v < 0.0 { v += 1.0; }
    let w = tx_get_width(tex, tps);
    let x = (w as f32* u) as u32;
    let y = (tx_get_height(tex, tps) as f32 * v) as u32;
    (w, x, y)
}

// get colour from texture and uv
fn get_tex_col(tex: u32, uv: (f32, f32), tps: &[u32], ts: &[u8]) -> Vec3{
    let (w, x, y) = uv_to_xy(uv, tex, tps);
    tx_get_sample(tex, tps, ts, x, y, w).powed_scalar(GAMMA)
}
