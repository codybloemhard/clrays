use crate::vec3::Vec3;
use crate::consts::{ EPSILON, MAX_RENDER_DIST };
use crate::cpu::inter::Ray;

#[derive(Copy, Clone, Debug)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AABB { // 24 bytes
    pub min: Vec3, // Vec3: 12 bytes
    pub max: Vec3, // Vec3: 12 bytes
}

impl Default for AABB{
    fn default() -> Self{
        Self::new()
    }
}

impl AABB {
    pub fn new() -> Self{
        Self {
            min: Vec3 { x: f32::MAX, y: f32::MAX, z: f32::MAX, },
            max: Vec3 { x: f32::MIN, y: f32::MIN, z: f32::MIN, },
        }
    }

    #[inline]
    pub fn set_default(&mut self) {
        self.max.x = f32::MIN;
        self.max.y = f32::MIN;
        self.max.z = f32::MIN;
        self.min.x = f32::MAX;
        self.min.y = f32::MAX;
        self.min.z = f32::MAX;
    }

    pub fn from_point_radius(p: Vec3, r: f32) -> Self{
        Self{
            min: p.subed(Vec3::uni(r)),
            max: p.added(Vec3::uni(r)),
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
            min: Vec3::new(minx, miny, minz),
            max: Vec3::new(maxx, maxy, maxz),
        }
    }

    #[inline]
    pub fn combine(&mut self, other: Self){
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);
        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }

    #[inline]
    pub fn combine_vertex(&mut self, other: Vec3){
        self.min.x = self.min.x.min(other.x);
        self.min.y = self.min.y.min(other.y);
        self.min.z = self.min.z.min(other.z);
        self.max.x = self.max.x.max(other.x);
        self.max.y = self.max.y.max(other.y);
        self.max.z = self.max.z.max(other.z);
    }

    #[inline]
    pub fn overlap(self, other: Self) -> AABB {
        let aabb = AABB {
            min: Vec3 {
                x: self.min.x.max(other.min.x),
                y: self.min.y.max(other.min.y),
                z: self.min.z.max(other.min.z),
            },
            max: Vec3 {
                x: self.max.x.min(other.max.x),
                y: self.max.y.min(other.max.y),
                z: self.max.z.min(other.max.z),
            }
        };
        if aabb.surface_area() > 0.0 {
            aabb
        } else {
            AABB::new()
        }
    }

    #[inline]
    pub fn combined(mut self, other: Self) -> Self{
        self.combine(other);
        self
    }

    pub fn grow(&mut self, v: Vec3){
        self.min.sub(v);
        self.max.add(v);
    }

    pub fn grown(mut self, v: Vec3) -> Self{
        self.grow(v);
        self
    }

    #[inline]
    pub fn midpoint(&self) -> Vec3{
        self.lerp(0.5)
    }

    #[inline]
    pub fn lerp(&self, val: f32) -> Vec3{
        self.min.scaled(1.0 - val).added(self.max.scaled(val))
    }

    // intersection formula, including inv_dir and dir_is_neg taken from:
    //  [source](http://www.pbr-book.org/3ed-2018/Shapes/Basic_Shape_Interface.html#Bounds3::IntersectP)
    // however, it required non-trivial changes after spending significant time debugging
    pub fn intersection(&self, ray: Ray, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> f32 {
        let ss = [&self.min, &self.max];

        // check inside box
        let p0 = self.min.subed(ray.pos);
        let p1 = self.max.subed(ray.pos);
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
        let v = self.max.subed(self.min);
        v.x * v.y * v.z
    }

    #[inline]
    pub fn surface_area(self) -> f32{
        let v = self.max.subed(self.min);
        if v.x < 0.0 || v.y < 0.0 || v.z < 0.0 { 0.0 }
        else {
            v.x * v.y * 2.0 +
                v.x * v.z * 2.0 +
                v.y * v.z * 2.0
        }
    }

    #[inline]
    pub fn is_in(self, other: &AABB) -> bool{
        other.min.less_eq(self.min) && self.max.less_eq(other.max)
    }

    #[inline]
    pub fn is_equal(self, other: &AABB) -> bool{
        self.min.equal(&other.min) && self.max.equal(&other.max)
    }

    #[inline]
    pub fn corners(self) -> Vec<Vec3> {
        vec![
            Vec3 {x: self.min.x, y: self.min.y, z: self.min.z },
            Vec3 {x: self.min.x, y: self.min.y, z: self.max.z },
            Vec3 {x: self.min.x, y: self.max.y, z: self.min.z },
            Vec3 {x: self.min.x, y: self.max.y, z: self.max.z },
            Vec3 {x: self.max.x, y: self.min.y, z: self.min.z },
            Vec3 {x: self.max.x, y: self.min.y, z: self.max.z },
            Vec3 {x: self.max.x, y: self.max.y, z: self.min.z },
            Vec3 {x: self.max.x, y: self.max.y, z: self.max.z },
        ]
    }

    #[inline]
    pub fn contains_vertex(self, vertex: Vec3) -> bool {
        self.min.less_eq(vertex) && vertex.less_eq(self.max)
    }

    /// find intersection that the ray will have with the bounding box
    #[inline]
    pub fn ray_intersections(self, ray: Ray, max_t: f32) -> Vec<Vec3> {
        let mut intersections: Vec<Vec3> = vec![];
        // per split plane of aabb compute intersections with ray
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let dt = ray.dir.fake_arr(axis);
            let p = ray.pos.fake_arr(axis);
            for value in [self.min.fake_arr(axis), self.max.fake_arr(axis)] {
                if (p-value).abs() < EPSILON{ // intersection on aabb boundary plane
                    intersections.push(ray.pos);
                } else if dt.abs() > EPSILON { // ensure non zero for direction movement
                    let t = (value-p) / dt;
                    if t > 0.0 && t <= max_t {
                        intersections.push(ray.travel(t))
                    }
                }
            }
        }
        // filter intersections within domain
        intersections.into_iter().filter(|int| self.contains_vertex(*int)).collect()
    }

    #[inline]
    pub fn split_by(&mut self, axis: Axis, value: f32, side: usize) {
        if side == 0 {
            self.min.update_by_axis(axis, value);
        } else {
            self.max.update_by_axis(axis, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::aabb::AABB;
    use crate::cpu::inter::Ray;
    use crate::vec3::Vec3;

    #[test]
    fn ray_intersections() {
        let aabb = AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE
        };
        let ray = Ray {
            pos: Vec3::ONE.scaled(-1.0),
            dir: Vec3::ONE.scaled(1.0)
        };
        let intersections = aabb.ray_intersections(ray, 1.0);
        assert!(intersections.contains(&aabb.min));
        assert!(!intersections.contains(&aabb.max));
        assert_ne!(AABB::from_points(&intersections), aabb);
        assert!(AABB::from_points(&intersections).is_in(&aabb));

        let aabb = AABB {
            min: Vec3::ZERO,
            max: Vec3::ONE
        };
        let ray = Ray {
            pos: Vec3::ONE.scaled(-1.0),
            dir: Vec3::ONE.scaled(2.0)
        };
        let intersections = aabb.ray_intersections(ray, 1.0);
        assert!(intersections.contains(&aabb.min));
        assert!(intersections.contains(&aabb.max));
        assert_eq!(AABB::from_points(&intersections), aabb);
    }
}

