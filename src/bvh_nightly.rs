use crate::scene::{Scene, Either, Model, ModelIndex, MeshIndex};
use crate::cpu::inter::*;
use crate::aabb::*;
use crate::vec3::Vec3;
use crate::consts::{ EPSILON };
use crate::mesh::Mesh;

#[derive(Default)]
pub struct Bvh{
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub mesh: Mesh,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex{
    pub bound: AABB,
    left_first: u32,
    count: u32,
}

impl Bvh{
    pub fn get_prim_counts(&self, current: usize, vec: &mut Vec<usize>){
        if current >= self.vertices.len() { return; }
        let vs = &self.vertices;
        let v = vs[current];
        if v.count > 0{ // leaf
            vec.push(v.count as usize);
        } else { // vertex
            self.get_prim_counts(v.left_first as usize, vec);
            self.get_prim_counts(v.left_first as usize + 1, vec);
        }
    }

    // (aabb hits, depth)
    #[inline]
    pub fn intersect(&self, ray: Ray, hit: &mut RayHit) -> (usize, usize){

        fn internal_intersect(bvh: &Bvh, current: usize, ray: Ray, hit: &mut RayHit, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> (usize, usize){
            let vs = &bvh.vertices;
            let is = &bvh.indices;
            let v = vs[current];
            if v.count > 0{ // leaf
                for i in is.iter().skip(v.left_first as usize).take(v.count as usize).map(|i| *i as usize){
                    inter_triangle(ray, &bvh.mesh.triangles[i], hit);
                }
                (0, v.count as usize)
            } else { // vertex
                let nodes = [
                    v.left_first as usize,
                    v.left_first as usize + 1,
                ];

                let ts = [
                    vs[nodes[0]].bound.intersection(ray, inv_dir, dir_is_neg),
                    vs[nodes[1]].bound.intersection(ray, inv_dir, dir_is_neg),
                ];

                let order = if ts[0] <= ts[1] { [0, 1] } else { [1, 0] };
                let mut x1: (usize, usize) = (0, 0);
                let mut x2: (usize, usize) = (0, 0);

                if ts[order[0]] < hit.t {
                    x1 = internal_intersect(bvh, nodes[order[0]], ray, hit, inv_dir, dir_is_neg);
                    if ts[order[1]] < hit.t {
                        x2 = internal_intersect(bvh, nodes[order[1]], ray, hit, inv_dir, dir_is_neg);
                    }
                }
                (2 + x1.0 + x2.0, 1 + x1.1.max(x2.1))
            }
        }

        // future: transform ray
        // -> ray.transform(self.transform.inverted())

        // TODO: doesn't work anymore with bvh? Problem in texture code
        // for plane in &scene.planes { inter_plane(ray, plane, hit); }
        if self.vertices.is_empty() { return (0, 0); }
        let inv_dir = ray.inverted().dir;
        let dir_is_neg : [usize; 3] = ray.direction_negations();
        let result = internal_intersect(self, 0, ray, hit, inv_dir, dir_is_neg);
        result
    }

    // // (aabb hits, depth)
    // #[inline]
    // pub fn occluded(&self, ray: Ray, scene: &Scene, ldist: f32) -> bool{
    //     #[allow(clippy::too_many_arguments)]
    //     fn internal_intersect(bvh: &Bvh, current: usize, scene: &Scene, ray: Ray, hit: &mut RayHit, ldist: f32, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> bool{
    //         let vs = &bvh.vertices;
    //         let is = &bvh.indices;
    //         let v = vs[current];
    //         if v.count > 0{ // leaf
    //             for i in is.iter().skip(v.left_first as usize).take(v.count as usize).map(|i| *i as usize){
    //                 if match scene.either_sphere_or_triangle(i){
    //                     Either::Left(s) => dist_sphere(ray, s),
    //                     Either::Right(t) => dist_triangle(ray, t),
    //                 } < ldist { return true }
    //             }
    //             false
    //         } else { // vertex
    //             let nodes = [
    //                 v.left_first as usize,
    //                 v.left_first as usize + 1,
    //             ];
    //
    //             let ts = [
    //                 vs[nodes[0]].bound.intersection(ray, inv_dir, dir_is_neg),
    //                 vs[nodes[1]].bound.intersection(ray, inv_dir, dir_is_neg),
    //             ];
    //
    //             let order = if ts[0] <= ts[1] { [0, 1] } else { [1, 0] };
    //
    //             if ts[order[0]] < hit.t {
    //                 if internal_intersect(bvh, nodes[order[0]], scene, ray, hit, ldist, inv_dir, dir_is_neg){
    //                     return true;
    //                 }
    //                 if ts[order[1]] < hit.t {
    //                     internal_intersect(bvh, nodes[order[1]], scene, ray, hit, ldist, inv_dir, dir_is_neg)
    //                 } else {
    //                     false
    //                 }
    //             } else {
    //                 false
    //             }
    //         }
    //     }
    //     // TODO: doesn't work anymore with bvh? Problem in texture code
    //     // for plane in &scene.planes { inter_plane(ray, plane, hit); }
    //     let inv_dir = ray.inverted().dir;
    //     let dir_is_neg : [usize; 3] = ray.direction_negations();
    //     let mut hit = RayHit::NULL;
    //     internal_intersect(self, 0, scene, ray, &mut hit, ldist, inv_dir, dir_is_neg)
    // }

    pub fn from_mesh(mesh: &Mesh, bins: usize) -> Self{
        let mesh = Mesh::load_model(&mesh.name);
        let prims = mesh.triangles.len();
        if prims == 0 {
            return Self{ indices: vec![], vertices: vec![], mesh};
        }
        let mut is = (0..prims as u32).into_iter().collect::<Vec<_>>();
        let mut vs = vec![Vertex::default(); prims * 2];
        let bounds = (0..prims).into_iter().map(|i|
            AABB::from_points(&[mesh.triangles[i].a, mesh.triangles[i].b, mesh.triangles[i].c])
        ).collect::<Vec<_>>();
        let mut poolptr = 2;
        Self::subdivide(&bounds, &mut is, &mut vs, 0, &mut poolptr, 0, prims, bins);
        Self{
            indices: is,
            vertices: vs,
            mesh
        }
    }

    // pub fn from(scene: &Scene, bins: usize) -> Self{
    //     // let mut sub_bvhs = vec![];
    //     // for mesh in &scene.meshes {
    //     //     sub_bvhs.push(Self::from_mesh(mesh, bins));
    //     // }
    //     // let sub_bvhs : Vec<Self> = scene.meshes.into_iter().map(|mesh| Self::from_mesh(&mesh, bins)).collect();
    //     // sub_bvhs[0]
    //     Self::from_mesh(&scene.meshes[0], bins)
    //
    //     // let prims = scene.spheres.len() + scene.triangles.len();
    //     // if prims == 0 {
    //     //     return Self{ indices: vec![], vertices: vec![] };
    //     // }
    //     //
    //     // // compute bvh for models
    //     // let sub_bvhs = scene.meshes.into_iter().map(|mesh| Self::from_mesh(&mesh, bins)).collect();
    //     //
    //     // assert_eq!(sub_bvhs.len(), 1);
    //     //
    //     // // let mut is = (0..prims as u32).into_iter().collect::<Vec<_>>();
    //     // // let mut vs = vec![Vertex::default(); prims * 2];
    //     // // let bounds = (0..prims).into_iter().map(|i|
    //     // //     match scene.either_sphere_or_triangle(i){
    //     // //         Either::Left(sphere) => AABB::from_point_radius(sphere.pos, sphere.rad),
    //     // //         Either::Right(tri) => AABB::from_points(&[tri.a, tri.b, tri.c]),
    //     // //     }
    //     // // ).collect::<Vec<_>>();
    //     // // let mut poolptr = 2;
    //     //
    //     // // Self::subdivide(&bounds, &mut is, &mut vs, 0, &mut poolptr, 0, prims, bins);
    //     //
    //     // // vs = vs.into_iter().filter(|v| v.bound != AABB::default()).collect::<Vec<_>>();
    //     // // println!("{:#?}", vs);
    //     //
    //     // sub_bvhs[0]
    //     //
    //     // // Self{
    //     // //     indices: is,
    //     // //     vertices: vs,
    //     // // }
    // }

    #[allow(clippy::too_many_arguments)]
    fn subdivide(bounds: &[AABB], is: &mut[u32], vs: &mut[Vertex], current: usize, poolptr: &mut u32, first: usize, count: usize, bins: usize){
        let v = &mut vs[current];
        v.bound = Self::bound(&is[first..first + count], bounds);

        if count < 3 { // leaf
            v.left_first = first as u32; // first
            v.count = count as u32;
            return;
        }

        let l_count = Self::partition(bounds, is, v.bound, first, count, bins);

        if l_count == 0 || l_count == count{ // leaf
            v.left_first = first as u32; // first
            v.count = count as u32;
            return;
        }

        v.count = 0; // internal vertex, not a leaf
        v.left_first = *poolptr; // left = poolptr, right = poolptr + 1
        *poolptr += 2;
        let lf = v.left_first as usize;

        Self::subdivide(bounds, is, vs, lf, poolptr, first, l_count, bins);
        Self::subdivide(bounds, is, vs, lf + 1, poolptr, first + l_count, count - l_count, bins);
    }

    fn partition(bounds: &[AABB], is: &mut[u32], bound: AABB, first: usize, count: usize, bins: usize) -> usize{
        let (axis, split) = Self::sah_binned(bounds, &is[first..first + count], bound, bins);
        let mut a = first; // first
        let mut b = first + count - 1; // last
        while a <= b{
            if bounds[is[a] as usize].midpoint().fake_arr(axis) < split{
                a += 1;
            } else {
                is.swap(a, b);
                b -= 1;
            }
        }
        a - first
    }

    fn bound(is: &[u32], bounds: &[AABB]) -> AABB {
        let mut bound = AABB::default();
        for i in is{
            bound.combine(bounds[*i as usize]);
        }
        bound.grown(Vec3::EPSILON)
    }

    fn sah_binned(bounds: &[AABB], is: &[u32], top_bound: AABB, bins: usize) -> (Axis, f32) {
        let binsf = bins as f32;
        let diff = top_bound.max.subed(top_bound.min);
        let axis_valid = [diff.x > binsf * EPSILON, diff.y > binsf * EPSILON, diff.z > binsf * EPSILON];

        // precompute lerps
        let mut lerps = vec![Vec3::ZERO; bins];
        for (i, item) in lerps.iter_mut().enumerate(){
            *item = top_bound.lerp(i as f32 / binsf);
        }

        // compute best combination; minimal cost
        let mut best : (f32, Axis, f32) = (f32::MAX, Axis::X, 0.0); // (cost, axis, split)
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            if !axis_valid[axis.as_usize()] {
                continue;
            }

            // iterate over 12 bins
            for lerp in lerps.iter(){
                let (mut ls, mut rs) = (0, 0);
                let (mut lb, mut rb) = (AABB::default(), AABB::default());

                let split = lerp.fake_arr(axis);
                for i in is{
                    let bound = bounds[*i as usize];
                    if bound.midpoint().fake_arr(axis) < split{
                        ls += 1;
                        lb.combine(bound);
                    } else {
                        rs += 1;
                        rb.combine(bound);
                    }
                }

                // get cost
                let cost = 3.0 + 1.0 + lb.surface_area() * ls as f32 + 1.0 + rb.surface_area() * rs as f32;
                if cost < best.0 {
                    best = (cost, axis, split);
                }
            }
        }

        let (_, axis, split) = best;
        (axis, split)
    }
}

