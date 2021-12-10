use crate::scene::{ Scene, AABB, Either };
use crate::vec3::Vec3;
use crate::cpu::inter::*;

#[derive(Clone, PartialEq, Debug)]
pub struct Bvh{
    indices: Vec<u32>,
    vertices: Vec<Vertex>,
}

impl Bvh{
    pub fn intersect<'a>(&self, scene: &'a Scene, ray: Ray) -> RayHit<'a>{
        fn inter<'a>(scene: &'a Scene, vs: &[Vertex], is: &[u32], ray: Ray, hit: &mut RayHit<'a>, current: usize) {
            let v = vs[current];
            if v.count == 0{ // vertex
                if hit_aabb(ray, v.bound).is_some(){
                    inter(scene, vs, is, ray, hit, v.left_first as usize);
                    inter(scene, vs, is, ray, hit, v.left_first as usize + 1);
                }
            } else { // leaf
                for i in is.iter().skip(v.left_first as usize).take(v.count as usize).map(|i| *i as usize){
                    match scene.either_sphere_or_triangle(i){
                        Either::Left(s) => inter_sphere(ray, s, hit),
                        Either::Right(t) => inter_triangle(ray, t, hit),
                    }
                }
            }
        }
        let mut closest = RayHit::NULL;
        inter(scene, &self.vertices, &self.indices, ray, &mut closest, 0);
        closest
    }

    pub fn occluded(&self, scene: &Scene, ray: Ray, ldist: f32) -> bool{
        fn inter<'a>(scene: &'a Scene, vs: &[Vertex], is: &[u32], ray: Ray, ldist: f32, current: usize) -> bool{
            let v = vs[current];
            if v.count == 0{ // vertex
                if hit_aabb(ray, v.bound).is_some(){
                    inter(scene, vs, is, ray, ldist, v.left_first as usize);
                    inter(scene, vs, is, ray, ldist, v.left_first as usize + 1);
                }
            } else { // leaf
                for i in is.iter().skip(v.left_first as usize).take(v.count as usize).map(|i| *i as usize){
                    if match scene.either_sphere_or_triangle(i){
                        Either::Left(s) => dist_sphere(ray, s),
                        Either::Right(t) => dist_triangle(ray, t),
                    } < ldist { return true }
                }
            }
            false
        }
        inter(scene, &self.vertices, &self.indices, ray, ldist, 0)
    }

    pub fn debug_intersect(&self, scene: &Scene, ray: Ray) -> usize{
        fn inter(scene: &Scene, vs: &[Vertex], is: &[u32], ray: Ray, current: usize, w: usize) -> usize{
            let v = vs[current];
            // if hit_aabb(ray, v.bound).is_none() { return 0; }
            let x = if hit_aabb(ray, v.bound).is_none() { 0 } else { 1 };
            if v.count == 0{ // vertex
                w*x +
                    inter(scene, vs, is, ray, v.left_first as usize, w) +
                    inter(scene, vs, is, ray, v.left_first as usize + 1, w)
            } else { // leaf
                let mut hit = RayHit::NULL;
                for i in is.iter().skip(v.left_first as usize).take(v.count as usize).map(|i| *i as usize){
                    match scene.either_sphere_or_triangle(i){
                        Either::Left(s) => inter_sphere(ray, s, &mut hit),
                        Either::Right(t) => inter_triangle(ray, t, &mut hit),
                    };
                }
                if hit.is_null() {
                    0
                } else {
                    w
                }
            }
        }
        inter(scene, &self.vertices, &self.indices, ray, 0, 1)
    }

    pub fn from(scene: &Scene) -> Self{
        let prims = scene.spheres.len() + scene.triangles.len();

        let mut is = (0..prims as u32).into_iter().collect::<Vec<_>>();
        let mut vs = vec![Vertex::default(); prims * 2 - 1];
        let mut poolptr = 2;

        Self::subdivide(scene, &mut is, &mut vs, 0, &mut poolptr, 0, prims);

        // vs = vs.into_iter().filter(|v| v.bound != AABB::default()).collect::<Vec<_>>();
        // println!("{:#?}", vs);

        Self{
            indices: is,
            vertices: vs,
        }
    }

    fn subdivide(scene: &Scene, is: &mut[u32], vs: &mut[Vertex], current: usize, poolptr: &mut u32, first: usize, count: usize){
        let v = &mut vs[current];
        v.bound = Self::bound(scene, first, count);

        if count < 3 { // leaf
            v.left_first = first as u32; // first
            v.count = count as u32;
            return;
        }

        v.left_first = *poolptr; // left = poolptr, right = poolptr + 1
        *poolptr += 2;

        let l_count = Self::partition(scene, is, v.bound, first, count);

        if l_count == 0 || l_count == count{ // leaf
            v.left_first = first as u32; // first
            v.count = count as u32;
            return;
        }

        v.count = 0; // internal vertex, not a leaf
        let lf = v.left_first as usize;

        Self::subdivide(scene, is, vs, lf, poolptr, first, l_count);
        Self::subdivide(scene, is, vs, lf + 1, poolptr, first + l_count, count - l_count);
    }

    fn partition(scene: &Scene, is: &mut[u32], bound: AABB, first: usize, count: usize) -> usize{
        fn is_left(index: usize, scene: &Scene, (plane, axis): (f32, Vec3)) -> bool{
            let plane_vec = axis.scaled(plane);
            match scene.either_sphere_or_triangle(index){
                Either::Left(sphere) => sphere.pos.muled(axis).less_eq(plane_vec),
                Either::Right(tri) => {
                    tri.a.muled(axis).less_eq(plane_vec) ||
                    tri.b.muled(axis).less_eq(plane_vec) ||
                    tri.c.muled(axis).less_eq(plane_vec)
                },
            }
        }
        let plane_axis = bound.midpoint_split();
        let mut a = first; // first
        let mut b = first + count - 1; // last
        while a < b{
            if is_left(is[a] as usize, scene, plane_axis){
                a += 1;
            } else {
                is.swap(a, b);
                b -= 1;
            }
        }
        a.min(count)
    }

    fn bound(scene: &Scene, first: usize, count: usize) -> AABB {
        let mut bound = AABB::default();
        for i in first..first + count{
            bound = match scene.either_sphere_or_triangle(i){
                Either::Left(s) => bound.combined(AABB::from_point_radius(s.pos, s.rad)),
                Either::Right(t) => bound.combined(AABB::from_points(&[t.a, t.b, t.c])),
            }
        }
        bound
    }
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Vertex{
    bound: AABB,
    left_first: u32,
    count: u32,
}
