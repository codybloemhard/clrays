use crate::scene::{ Scene, Either };
use crate::cpu::inter::*;
use crate::aabb::*;
use crate::vec3::Vec3;
use crate::consts::{ EPSILON };

#[derive(Clone, Debug)]
pub struct Bvh{
    indices: Vec<u32>,
    vertices: Vec<Vertex>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex{
    bound: AABB,
    left_first: u32,
    count: u32,
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
        let bounds = (0..prims).into_iter().map(|i|
            match scene.either_sphere_or_triangle(i){
                Either::Left(sphere) => AABB::from_point_radius(sphere.pos, sphere.rad),
                Either::Right(tri) => AABB::from_points(&[tri.a, tri.b, tri.c]),
            }
        ).collect::<Vec<_>>();
        let mut poolptr = 2;

        Self::subdivide(&bounds, &mut is, &mut vs, 0, &mut poolptr, 0, prims);

        // vs = vs.into_iter().filter(|v| v.bound != AABB::default()).collect::<Vec<_>>();
        // println!("{:#?}", vs);

        Self{
            indices: is,
            vertices: vs,
        }
    }

    fn subdivide(bounds: &[AABB], is: &mut[u32], vs: &mut[Vertex], current: usize, poolptr: &mut u32, first: usize, count: usize){
        let v = &mut vs[current];
        v.bound = Self::bound(&is[first..first + count], bounds);

        if count < 3 { // leaf
            v.left_first = first as u32; // first
            v.count = count as u32;
            return;
        }

        v.left_first = *poolptr; // left = poolptr, right = poolptr + 1
        *poolptr += 2;

        let l_count = Self::partition(bounds, is, v.bound, first, count);

        if l_count == 0 || l_count == count{ // leaf
            v.left_first = first as u32; // first
            v.count = count as u32;
            return;
        }

        v.count = 0; // internal vertex, not a leaf
        let lf = v.left_first as usize;

        Self::subdivide(bounds, is, vs, lf, poolptr, first, l_count);
        Self::subdivide(bounds, is, vs, lf + 1, poolptr, first + l_count, count - l_count);
    }

    fn partition(_bounds: &[AABB], _is: &mut[u32], _bound: AABB, _first: usize, _count: usize) -> usize{
        // fn is_left(index: usize, scene: &Scene, (plane, axis): (f32, Vec3)) -> bool{
        //     let plane_vec = axis.scaled(plane);
        //     match scene.either_sphere_or_triangle(index){
        //         Either::Left(sphere) => sphere.pos.muled(axis).less_eq(plane_vec),
        //         Either::Right(tri) => {
        //             tri.a.muled(axis).less_eq(plane_vec) ||
        //             tri.b.muled(axis).less_eq(plane_vec) ||
        //             tri.c.muled(axis).less_eq(plane_vec)
        //         },
        //     }
        // }
        // let plane_axis = bound.midpoint_split();
        // let mut a = first; // first
        // let mut b = first + count - 1; // last
        // while a < b{
        //     if is_left(is[a] as usize, scene, plane_axis){
        //         a += 1;
        //     } else {
        //         is.swap(a, b);
        //         b -= 1;
        //     }
        // }
        // a.min(count)
        0
    }

    fn bound(is: &[u32], bounds: &[AABB]) -> AABB {
        let mut bound = AABB::default();
        for i in is{
            bound.combine(bounds[*i as usize]);
        }
        bound
    }

    fn sah_binned(bounds: &[AABB], bound: AABB) -> (Axis, f32) {
        let diff = bound.b.subed(bound.a);
        let axis_valid = [diff.x > 12.0 * EPSILON, diff.y > 12.0 * EPSILON, diff.z > 12.0 * EPSILON];

        // precompute lerps
        let mut lerps = [Vec3::ZERO; 12];
        for (i, item) in lerps.iter_mut().enumerate(){
            *item = bound.lerp(i as f32 / 12.0);
        }

        // precompute midpoints
        // let mid_points: Vec<Vec3> = bounds.iter().map(|b| b.midpoint()).collect();
        // let n = mid_points.len();

        // compute best combination; minimal cost
        let mut best : (f32, Axis, usize) = (f32::MAX, Axis::X, 0); // (cost, axis, i_lerp)
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            if !axis_valid[axis.as_usize()] {
                continue;
            }

            // let mut index_reshuffled = vec![0; n];
            // for (i, item) in index_reshuffled.iter_mut().enumerate().take(n){
            //     *item = i;
            // }

            // sort primitives for axis with quicksort O(n*log(n))
            // let vals: Vec<f32> = mid_points.iter().map(|point| point.fake_arr(axis)).collect();
            // crate::bvh::my_little_sorter(&vals, &mut index_reshuffled);

            // let mut sides: [Vec<u32>; 2] = [vec![], vec![]];
            // let mut side_bounds: [AABB; 2] = [AABB::new(); 2];
            // for i in 0..n{
            //     // initially all prims on the right
            //     // push them in reverse, so we can then easily pop later on
            //     sides[1].push(&primitives[index_reshuffled[n - i - 1]]);
            // }
            // let mut i_split: usize = 0;

            let (mut ls, mut rs) = (0, 0);
            let (mut lb, mut rb) = (AABB::default(), AABB::default());

            // iterate over 12 bins
            for (i_lerp, lerp) in lerps.iter().enumerate(){
                let split = lerp.fake_arr(axis);
                for bound in bounds.iter(){
                    if bound.midpoint().fake_arr(axis) < split{
                        ls += 1;
                        lb.combine(*bound);
                    } else {
                        rs += 1;
                        rb.combine(*bound);
                    }
                }

                // place over prims from right split to left split
                // while i_split < n && mid_points[index_reshuffled[i_split]].fake_arr(axis) < tmp{
                //     let prim = sides[1].pop().unwrap();
                //     sides[0].push(prim);
                //
                //     side_bounds[0].a.x = side_bounds[0].a.x.min(prim.bound.a.x);
                //     side_bounds[0].a.y = side_bounds[0].a.y.min(prim.bound.a.y);
                //     side_bounds[0].a.z = side_bounds[0].a.z.min(prim.bound.a.z);
                //
                //     side_bounds[0].b.x = side_bounds[0].b.x.max(prim.bound.b.x);
                //     side_bounds[0].b.y = side_bounds[0].b.y.max(prim.bound.b.y);
                //     side_bounds[0].b.z = side_bounds[0].b.z.max(prim.bound.b.z);
                //
                //     i_split += 1;
                // }

                // recompute bound for right
                // side_bounds[1] = AABB::new();
                // for prim in sides[1].iter() {
                //     side_bounds[1].a.x = side_bounds[1].a.x.min( prim.bound.a.x );
                //     side_bounds[1].a.y = side_bounds[1].a.y.min( prim.bound.a.y );
                //     side_bounds[1].a.z = side_bounds[1].a.z.min( prim.bound.a.z );
                //
                //     side_bounds[1].b.x = side_bounds[1].b.x.max( prim.bound.b.x );
                //     side_bounds[1].b.y = side_bounds[1].b.y.max( prim.bound.b.y );
                //     side_bounds[1].b.z = side_bounds[1].b.z.max( prim.bound.b.z );
                // }

                // get cost
                let cost = 3.0 + 1.0 + lb.surface_area() * ls as f32 + 1.0 + rb.surface_area() * rs as f32;
                if cost < best.0 {
                    best = (cost, axis, i_lerp);
                }
            }
        }

        // apply best
        let (_, axis, i_lerp) = best;
        let val = lerps[i_lerp].fake_arr(axis);
        (axis, val)

        // Define primitives for left and right
        // for i in 0..n {
        //     if mid_points[i].fake_arr(axis) < val {
        //         left.push(primitives[i]);
        //     } else {
        //         right.push(primitives[i]);
        //     }
        // }

        // TODO: improve cost function
        // decide whether we apply the primitives into a leaf node
        // if left.is_empty() || right.is_empty() => leaf
    }
}

