use crate::vec3::Vec3;
use crate::consts::{ EPSILON, MAX_RENDER_DIST };
use crate::cpu::inter::Ray;

#[derive(Copy, Clone, Debug)]
pub enum Axis {
    X,
    Y,
    Z
}

impl Axis {
    pub fn as_usize(&self) -> usize{
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct AABB { // 24 bytes
    pub a: Vec3, // Vec3: 12 bytes
    pub b: Vec3, // Vec3: 12 bytes
}

impl Default for AABB{
    fn default() -> Self{
        Self::new()
    }
}

impl AABB {
    pub fn new() -> Self{
        Self {
            a: Vec3 { x: f32::MAX, y: f32::MAX, z: f32::MAX, },
            b: Vec3 { x: f32::MIN, y: f32::MIN, z: f32::MIN, },
        }
    }

    pub fn from_point_radius(p: Vec3, r: f32) -> Self{
        Self{
            a: p.subed(Vec3::uni(r)),
            b: p.added(Vec3::uni(r)),
        }
    }

    pub fn from_points(ps: &[Vec3]) -> Self{
        let (mut minx, mut miny, mut minz): (f32, f32, f32) = (f32::MAX, f32::MAX, f32::MAX);
        let (mut maxx, mut maxy, mut maxz): (f32, f32, f32) = (f32::MIN, f32::MIN, f32::MIN);

        for p in ps{
            minx = minx.min(p.x);
            miny = miny.min(p.y);
            minz = minz.min(p.z);
            maxx = maxx.max(p.x);
            maxy = maxy.max(p.y);
            maxz = maxz.max(p.z);
        }

        Self{
            a: Vec3::new(minx, miny, minz),
            b: Vec3::new(maxx, maxy, maxz),
        }
    }

    #[inline]
    pub fn combine(&mut self, other: Self){
        self.a.x = self.a.x.min(other.a.x);
        self.a.y = self.a.y.min(other.a.y);
        self.a.z = self.a.z.min(other.a.z);
        self.b.x = self.b.x.max(other.b.x);
        self.b.y = self.b.y.max(other.b.y);
        self.b.z = self.b.z.max(other.b.z);
    }

    #[inline]
    pub fn combined(mut self, other: Self) -> Self{
        self.combine(other);
        self
    }

    pub fn grow(&mut self, v: Vec3){
        self.a.sub(v);
        self.b.add(v);
    }

    pub fn grown(mut self, v: Vec3) -> Self{
        self.grow(v);
        self
    }

    pub fn midpoint(&self) -> Vec3{
        self.lerp(0.5)
    }

    pub fn lerp(&self, val: f32) -> Vec3{
        self.a.scaled(1.0 - val).added(self.b.scaled(val))
    }

    // intersection formula, including inv_dir and dir_is_neg taken from:
    //  [source](http://www.pbr-book.org/3ed-2018/Shapes/Basic_Shape_Interface.html#Bounds3::IntersectP)
    // however, it required such non-trivial changes after spending significant time debugging
    // I feel like calling this code my own
    pub fn intersection(&self, ray: Ray, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> f32 {
        let ss = [&self.a, &self.b];

        // check inside box
        let p0 = self.a.subed(ray.pos);
        let p1 = self.b.subed(ray.pos);
        let inside = [
            p0.x <= EPSILON && p1.x >= -EPSILON,
            p0.y <= EPSILON && p1.y >= -EPSILON,
            p0.z <= EPSILON && p1.z >= -EPSILON
        ];
        // println!("{:?}", inside);
        if inside[0] && inside[1] && inside[2] { return 0.0; };

        let mut tmin = -MAX_RENDER_DIST;
        let mut tmax = MAX_RENDER_DIST;
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let i = axis.as_usize();
            if inv_dir.fake_arr(axis).abs() == f32::INFINITY && inside[i] { continue; }

            let t0 = (ss[  dir_is_neg[i]].fake_arr(axis) - ray.pos.fake_arr(axis)) * inv_dir.fake_arr(axis);
            let t1 = (ss[1-dir_is_neg[i]].fake_arr(axis) - ray.pos.fake_arr(axis)) * inv_dir.fake_arr(axis);
            if t0 > tmax || t1 < tmin { return MAX_RENDER_DIST; }

            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
        }
        if tmin <= 0.0 && tmax >= 0.0 { 0.0 } else if tmin >= 0.0 { tmin } else { tmax }
    }

    pub fn volume(self) -> f32{
        let v = self.b.subed(self.a);
        v.x * v.y * v.z
    }

    pub fn surface_area(self) -> f32{
        let v = self.b.subed(self.a);
        v.x * v.y * 2.0 +
        v.x * v.z * 2.0 +
        v.y * v.z * 2.0
    }
}
