use crate::scene::{Scene, Either, Model, ModelIndex, MeshIndex, Triangle};
use crate::cpu::inter::*;
use crate::aabb::*;
use crate::vec3::Vec3;
use crate::consts::{ EPSILON };
use crate::mesh::Mesh;
use std::sync::Arc;
use stopwatch::Stopwatch;

#[derive(Default)]
pub struct Bvh{
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub mesh: Mesh
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
    pub fn intersect(&self, ray: Ray, scene: &Scene, hit: &mut RayHit) -> (usize, usize){

        fn internal_intersect(bvh: &Bvh, current: usize, ray: Ray, scene: &Scene, hit: &mut RayHit, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> (usize, usize){
            let vs = &bvh.vertices;
            let is = &bvh.indices;
            let v = vs[current];
            if v.count > 0{ // leaf
                for i in is.iter().skip(v.left_first as usize).take(v.count as usize).map(|i| *i as usize){
                    inter_triangle(ray,scene.get_mesh_triangle(&bvh.mesh, i), hit);
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
                    x1 = internal_intersect(bvh, nodes[order[0]], ray, scene, hit, inv_dir, dir_is_neg);
                    if ts[order[1]] < hit.t {
                        x2 = internal_intersect(bvh, nodes[order[1]], ray, scene, hit, inv_dir, dir_is_neg);
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
        let result = internal_intersect(self, 0, ray, scene, hit, inv_dir, dir_is_neg);
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

    pub fn from_mesh(mesh: Mesh, triangles: &Vec<Triangle>, bins: usize) -> Self{
        let n = triangles.len();
        if n == 0 {
            return Self{ indices: vec![], vertices: vec![], mesh };
        }
        let mut is = (0..n as u32).into_iter().collect::<Vec<_>>();
        let mut vs = vec![Vertex::default(); n * 2];
        println!("Generating bounds...");
        let watch = Stopwatch::start_new();
        let bounds = (0..n).into_iter().map(|i|
            AABB::from_points(&[triangles[i].a, triangles[i].b, triangles[i].c])
        ).collect::<Vec<_>>();
        let mut elapsed = watch.elapsed_ms();
        println!("done building bounds in {}...", elapsed);
        println!("Generating midpoints...");
        let midpoints = (0..n).into_iter().map(|i|
            bounds[i].midpoint().as_array()
        ).collect::<Vec<_>>();
        let mut elapsed = watch.elapsed_ms();
        println!("done building midpoints in {}...", elapsed);
        let mut poolptr = 2;
        Self::subdivide(&bounds, &midpoints, &mut is, &mut vs, 0, &mut poolptr, 0, n, bins);
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
    fn subdivide(bounds: &[AABB], midpoints: &[[f32;3]], is: &mut[u32], vs: &mut[Vertex], current: usize, poolptr: &mut u32, first: usize, count: usize, bins: usize){
        let v = &mut vs[current];
        v.bound = Self::union_bound(&is[first..first + count], bounds);

        if count < 3 { // leaf
            v.left_first = first as u32; // first
            v.count = count as u32;
            return;
        }

        let l_count = Self::partition(bounds, midpoints, is, v.bound, first, count, bins);

        if l_count == 0 || l_count == count{ // leaf
            v.left_first = first as u32; // first
            v.count = count as u32;
            return;
        }

        v.count = 0; // internal vertex, not a leaf
        v.left_first = *poolptr; // left = poolptr, right = poolptr + 1
        *poolptr += 2;
        let lf = v.left_first as usize;

        Self::subdivide(bounds, midpoints, is, vs, lf, poolptr, first, l_count, bins);
        Self::subdivide(bounds, midpoints, is, vs, lf + 1, poolptr, first + l_count, count - l_count, bins);
    }

    fn partition(bounds: &[AABB], midpoints: &[[f32;3]], is: &mut[u32], bound: AABB, first: usize, count: usize, bins: usize) -> usize{
        let (axis, split) = Self::sah_binned(bounds, midpoints, &is[first..first + count], bound, bins);
        let mut a = first; // first
        let mut b = first + count - 1; // last
        let u = axis.as_usize();
        while a <= b{
            if midpoints[is[a] as usize][u] < split{
                a += 1;
            } else {
                is.swap(a, b);
                b -= 1;
            }
        }
        a - first
    }

    fn union_bound(is: &[u32], bounds: &[AABB]) -> AABB {
        let mut bound = AABB::default();
        for i in is{
            bound.combine(bounds[*i as usize]);
        }
        bound.grown(Vec3::EPSILON)
    }

    fn sah_binned(bounds: &[AABB], midpoints: &[[f32;3]], is: &[u32], top_bound: AABB, bins: usize) -> (Axis, f32) {
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
            let u = axis.as_usize();
            if !axis_valid[u] { continue; }
            let k1 = (bins as f32*(1.0-EPSILON))/(top_bound.max.fake_arr(axis)-top_bound.min.fake_arr(axis));
            let k0 = top_bound.min.fake_arr(axis);

            // place bounds in bins
            let mut sep = vec![vec![];bins];
            for i in is {
                let midpoint = midpoints[*i as usize];
                let index = k1*(midpoint[u]-k0);
                if index as usize >= bins {
                    println!("topbounds: {:?}\n, bound: {:?}\n, midpoint: {:?}\n, axis: {:?}", top_bound, bounds[*i  as usize], midpoint, axis);
                    println!("{}, {}", index, index as usize);
                }
                sep[index as usize].push(i);
            }
            // generate bounds of bins
            let mut binbounds = vec![AABB::new();bins];
            let mut bincounts = vec![0;bins];
            for bin_index in 0..bins {
                for bound_index in sep[bin_index]{
                    binbounds[bin_index].combine(bounds[*bound_index as usize]);
                }
                bincounts[bin_index] = sep[bin_index].len();
            }

            // iterate over bins
            for (lerp_index,lerp) in lerps.iter().enumerate(){

                let split = lerp.fake_arr(axis);
                // construct bounds
                let (mut ls, mut rs) = (0, 0);
                let (mut lb, mut rb) = (AABB::default(), AABB::default());
                for j in 0..lerp_index { // left of split
                    ls += bincounts[j];
                    lb.combine(binbounds[j]);
                }
                for j in 0..lerp_index { // right of split
                    rs += bincounts[j];
                    rb.combine(binbounds[j]);
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

