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
    left_first: usize,
    count: usize,
}

pub struct BuilderData {
    bounds: Vec<AABB>,
    midpoints: Vec<[f32; 3]>,
    is: Vec<usize>,
    vs: Vec<Vertex>,
    bins: usize,
    watch: Stopwatch,
    times: Vec<i64>,
    info: Info,
}

struct Info {
    maxdepth: usize,
    counter: usize,
}

impl Bvh{
    #[allow(clippy::too_many_arguments)]
    fn subdivide(data: &mut BuilderData, current: usize, poolptr: &mut usize, first: usize, count: usize, depth: usize){
        let midpoints = &data.midpoints;
        let bins = data.bins;
        let binsf = bins as f32;
        let mut v = &mut data.vs[current];
        data.watch.start();
        let sub_is = &data.is[first..first + count];
        let top_bound = Self::union_bound(sub_is, &data.bounds);
        data.times[0] += data.watch.elapsed_ms();

        if count < 3 { // leaf
            v.bound = top_bound;
            v.left_first = first; // first
            v.count = count;
            return;
        }

        data.watch.start();
        // sah binned
        let diff = top_bound.max.subed(top_bound.min);
        let axis_valid = [diff.x > binsf * EPSILON, diff.y > binsf * EPSILON, diff.z > binsf * EPSILON];

        // precompute lerps
        let mut lerps = vec![Vec3::ZERO; bins];
        for (i, item) in lerps.iter_mut().enumerate(){
            *item = top_bound.lerp(i as f32 / binsf);
        }

        // compute best combination; minimal cost
        let (mut ls, mut rs) = (0, 0);
        let (mut lb, mut rb) = (AABB::default(), AABB::default());
        let mut best_aabb_left = AABB::default();
        let mut best_aabb_right = AABB::default();
        let mut best_cost = f32::MAX;
        let mut best_axis = Axis::X;
        let mut best_split = 0.0;
        data.times[3] += data.watch.elapsed_ms();

        data.watch.start();
        for axis in [Axis::X, Axis::Y, Axis::Z] {

            let u = axis.as_usize();
            if !axis_valid[u] {
                continue;
            }
            let k1 = (binsf*(1.0-EPSILON))/(top_bound.max.fake_arr(axis)-top_bound.min.fake_arr(axis));
            let k0 = top_bound.min.fake_arr(axis);

            // place bounds in bins
            // generate bounds of bins
            let mut binbounds = vec![AABB::new();bins];
            let mut bincounts : Vec<usize> = vec![0;bins];
            let mut index: usize ;
            for index_triangle in sub_is {
                index = (k1*(midpoints[*index_triangle][u]-k0)) as usize;
                binbounds[index].combine(data.bounds[*index_triangle]);
                bincounts[index] += 1;
            }

            // iterate over bins
            for (lerp_index,lerp) in lerps.iter().enumerate(){
                let split = lerp.fake_arr(axis);
                // reset values
                ls = 0; rs = 0;
                lb.set_default(); rb.set_default();
                // construct bounds
                for j in 0..lerp_index { // left of split
                    ls += bincounts[j];
                    lb.combine(binbounds[j]);
                }
                for j in lerp_index..bins { // right of split
                    rs += bincounts[j];
                    rb.combine(binbounds[j]);
                }

                // get cost
                let cost = 3.0 + 1.0 + lb.surface_area() * ls as f32 + 1.0 + rb.surface_area() * rs as f32;
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = split;
                }
            }
        }
        data.times[1] += data.watch.elapsed_ms();

        // partition
        data.watch.start();
        let mut a = first; // first
        let mut b = first + count - 1; // last
        let u = best_axis.as_usize();
        while a <= b{
            if data.midpoints[data.is[a]][u] < best_split{
                a += 1;
            } else {
                data.is.swap(a, b);
                b -= 1;
            }
        }
        let l_count = a - first;
        data.times[2] += data.watch.elapsed_ms();

        if l_count == 0 || l_count == count{ // leaf
            v.bound = top_bound;
            v.left_first = first; // first
            v.count = count;
            return;
        }

        v.bound = top_bound;
        v.count = 0; // internal vertex, not a leaf
        v.left_first = *poolptr; // left = poolptr, right = poolptr + 1
        *poolptr += 2;
        let lf = v.left_first;


        Self::subdivide(data, lf, poolptr, first, l_count, depth + 1);
        Self::subdivide(data, lf + 1, poolptr, first + l_count, count - l_count, depth + 1);
    }

    fn union_bound(is: &[usize], bounds: &[AABB]) -> AABB {
        let mut bound = AABB::default();
        for i in is{
            bound.combine(bounds[*i]);
        }
        bound.grown(Vec3::EPSILON)
    }

    pub fn get_prim_counts(&self, current: usize, vec: &mut Vec<usize>){
        if current >= self.vertices.len() { return; }
        let vs = &self.vertices;
        let v = vs[current];
        if v.count > 0{ // leaf
            vec.push(v.count);
        } else { // vertex
            self.get_prim_counts(v.left_first, vec);
            self.get_prim_counts(v.left_first + 1, vec);
        }
    }

    pub fn intersect(&self, ray: Ray, scene: &Scene, hit: &mut RayHit) -> (usize, usize){ (0,0) }
    pub fn from_mesh(mesh: Mesh, triangles: &Vec<Triangle>, bins: usize) -> Self{
        let n = triangles.len();
        if n == 0 {
            return Self{ indices: vec![], vertices: vec![], mesh };
        }
        let mut is = (0..n).into_iter().collect::<Vec<_>>();
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
        let mut data = BuilderData {
            bounds,
            midpoints,
            is,
            vs,
            bins,
            watch,
            times: vec![0,0,0,0,0],
            info: Info {
                maxdepth: 0,
                counter: 0,
            }
        };
        let mut poolptr = 2;
        let watch = Stopwatch::start_new();
        println!("{:?}", data.times);
        println!("{:?}", watch.elapsed_ms());
        Self::subdivide(&mut data, 0, &mut poolptr, 0, n, 0);

        let p = data.is.into_iter().map(|i| i as u32).collect();
        Self{
            indices: p,
            vertices: data.vs,
            mesh
        }
    }
}

