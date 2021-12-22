use crate::scene::{ Sphere, Plane, Triangle, MaterialIndex };
use crate::vec3::Vec3;
use crate::aabb::AABB;
use crate::consts::*;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Ray{
    pub pos: Vec3,
    pub dir: Vec3,
}

impl Ray{
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
    pub mat: MaterialIndex,
    pub uvtype: u8,
    pub sphere: Option<&'a Sphere>,
}

impl RayHit<'_>{
    pub const NULL: Self = RayHit{
        pos: Vec3::ZERO,
        nor: Vec3::ZERO,
        t: MAX_RENDER_DIST,
        mat: 0,
        uvtype: 255,
        sphere: None,
    };

    #[inline]
    pub fn is_null(&self) -> bool{
        self.uvtype == 255
    }
}

// ray-sphere intersection
#[inline]
pub fn inter_sphere<'a>(ray: Ray, sphere: &'a Sphere, closest: &mut RayHit<'a>){
    let l = Vec3::subed(sphere.pos, ray.pos);
    let tca = Vec3::dot(ray.dir, l);
    let d = tca * tca - Vec3::dot(l, l) + sphere.rad * sphere.rad;
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
    closest.mat = sphere.mat;
    closest.uvtype = UV_SPHERE;
    closest.sphere = Some(sphere);
}

#[inline]
pub fn dist_sphere(ray: Ray, sphere: &Sphere) -> f32{
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

// ray-plane intersection
#[inline]
pub fn inter_plane<'a>(ray: Ray, plane: &'a Plane, closest: &mut RayHit<'a>){
    let divisor = Vec3::dot(ray.dir, plane.nor);
    if divisor.abs() < EPSILON { return; }
    let planevec = Vec3::subed(plane.pos, ray.pos);
    let t = Vec3::dot(planevec, plane.nor) / divisor;
    if t < EPSILON { return; }
    if t > closest.t { return; }
    closest.t = t;
    closest.pos = ray.pos.added(ray.dir.scaled(t));
    closest.nor = plane.nor;
    closest.mat = plane.mat;
    closest.uvtype = UV_PLANE;
    closest.sphere = None;
}

#[inline]
pub fn dist_plane(ray: Ray, plane: &Plane) -> f32{
    let divisor = Vec3::dot(ray.dir, plane.nor);
    if divisor.abs() < EPSILON { return MAX_RENDER_DIST; }
    let planevec = Vec3::subed(plane.pos, ray.pos);
    let t = Vec3::dot(planevec, plane.nor) / divisor;
    if t < EPSILON { return MAX_RENDER_DIST; }
    t
}

// ray-triangle intersection
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm?oldformat=true
#[inline]
#[allow(clippy::many_single_char_names)]
pub fn inter_triangle<'a>(ray: Ray, tri: &'a Triangle, closest: &mut RayHit<'a>){
    let edge1 = Vec3::subed(tri.b, tri.a);
    let edge2 = Vec3::subed(tri.c, tri.a);
    let h = Vec3::crossed(ray.dir, edge2);
    let a = Vec3::dot(edge1, h);
    if a > -EPSILON * 0.01 && a < EPSILON * 0.01 { return; } // ray parallel to tri
    let f = 1.0 / a;
    let s = Vec3::subed(ray.pos, tri.a);
    let u = f * Vec3::dot(s, h);
    if !(0.0..=1.0).contains(&u) { return; }
    let q = Vec3::crossed(s, edge1);
    let v = f * Vec3::dot(ray.dir, q);
    if v < 0.0 || u + v > 1.0 { return; }
    let t = f * Vec3::dot(edge2, q);
    if t <= EPSILON { return; }
    if t > closest.t { return; }
    closest.t = t;
    closest.pos = ray.pos.added(ray.dir.scaled(t));
    closest.nor = Vec3::crossed(edge1, edge2).normalized_fast();
    closest.mat = tri.mat;
    closest.uvtype = UV_PLANE;
    closest.sphere = None;
}

#[inline]
#[allow(clippy::many_single_char_names)]
pub fn dist_triangle(ray: Ray, tri: &Triangle) -> f32{
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

#[inline]
pub fn hit_aabb(ray: Ray, aabb: AABB) -> Option<(f32, f32)>{
    let inv_dir = Vec3 {
        x: if ray.dir.x.abs() > EPSILON { 1.0 / ray.dir.x } else { f32::MAX },
        y: if ray.dir.y.abs() > EPSILON { 1.0 / ray.dir.y } else { f32::MAX },
        z: if ray.dir.z.abs() > EPSILON { 1.0 / ray.dir.z } else { f32::MAX },
    };
    let dir_is_neg : [bool; 3] = [
        ray.dir.x < 0.0,
        ray.dir.y < 0.0,
        ray.dir.z < 0.0,
    ];

    let mut t_min;
    let mut t_max;
    let ss = [&aabb.min, &aabb.max];

    // Compute intersections with x and y slabs.
    let tx_min = (ss[  dir_is_neg[0] as usize].x - ray.pos.x) * inv_dir.x;
    let tx_max = (ss[1-dir_is_neg[0] as usize].x - ray.pos.x) * inv_dir.x;
    let ty_min = (ss[  dir_is_neg[1] as usize].y - ray.pos.y) * inv_dir.y;
    let ty_max = (ss[1-dir_is_neg[1] as usize].y - ray.pos.y) * inv_dir.y;

    // Check intersection within x and y bounds.
    if (tx_min > ty_max) || (tx_max < ty_min) {
        return None;
    }
    t_min = if ty_min > tx_min { ty_min } else { tx_min };
    t_max = if ty_max < tx_max { ty_max } else { tx_max };

    // Compute intersections z slab.
    let tz_min = (ss[  dir_is_neg[2] as usize].z - ray.pos.z) * inv_dir.z;
    let tz_max = (ss[1-dir_is_neg[2] as usize].z - ray.pos.z) * inv_dir.z;

    // Check intersection within x and y and z bounds.
    if (t_min > tz_max) || (t_max < tz_min) {
        return None;
    }
    t_min = if tz_min > t_min { tz_min } else { t_min };
    t_max = if tz_max < t_max { tz_max } else { t_max };

    if tz_min > t_min { t_min = tz_min; }
    if tz_max < t_max { t_max = tz_max; }

    Some((t_min, t_max))
}
