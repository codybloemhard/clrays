use crate::scene::{Scene, Either, Model, ModelIndex, MeshIndex, Triangle};
use crate::cpu::inter::*;
use crate::aabb::*;
use crate::vec3::Vec3;
use crate::consts::{ EPSILON };
use crate::mesh::Mesh;

#[derive(Default)]
pub struct Bvh{
    pub vertices: Vec<Vertex>,
    pub mesh: Mesh
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex{
    pub bound: AABB,
    left_first: usize,
    count: usize,
}

impl Bvh{
    #[allow(clippy::too_many_arguments)]
    fn subdivide(bounds: &mut Vec<AABB>, vs: &mut Vec<Vertex>, bins: usize, triangles: &mut Vec<Triangle>){
        let current = 0;
        let first = 0;
        let mut poolptr = 2;
        let count = triangles.len();
        let mut depth = 0;

        let binsf = bins as f32;
        let binsf_inf = 1.0 / binsf;

        let mut stack = vec![]; // [(current,first,count,step)]
        stack.push(StackItem {current,first,count,depth});

        let mut lerps = vec![Vec3::ZERO; bins-1];
        let mut binbounds = vec![AABB::new();bins];
        let mut bincounts : Vec<usize> = vec![0;bins];
        let aabb_null = AABB::default();

        let (mut lb, mut rb) = (AABB::default(), AABB::default());

        let mut best_aabb_left = AABB::default();
        let mut best_aabb_right = AABB::default();
        let mut best_axis = Axis::X;
        let mut best_split = 0.0;

        let mut v : &mut Vertex = &mut Vertex::default();
        let mut top_bound;

        let mut current = 0;
        let mut first = 0;
        let mut count = 0;

        let mut depth_timers = vec![];
        let mut depth_items = vec![];

        while stack.len() > 0 {

            let mut x = stack.pop().unwrap();
            // println!("{:?}", x);
            // measure time in depth
            depth = x.depth;
            if depth >= depth_timers.len() {
                depth_timers.push(0);
                depth_items.push(0);
            }

            depth_items[depth] += x.count;
            current = x.current;
            count = x.count;
            first = x.first;
            // println!("{:?}", x);
            v = &mut vs[current];
            // sub_is = &is[first..first + count];
            // top_bound = union_bound(sub_is, bounds);
            let sub_range = first..first + count;
            top_bound = union_bound(&bounds[sub_range.clone()]);
            v.bound = top_bound;

            if count < 3 { // leaf
                v.left_first = first; // first
                v.count = count;
                continue;
            }

            // sah binned
            let diff = top_bound.max.subed(top_bound.min);
            let axis_valid = [diff.x > binsf * EPSILON, diff.y > binsf * EPSILON, diff.z > binsf * EPSILON];

            // precompute lerps
            for (i, item) in lerps.iter_mut().enumerate(){
                *item = top_bound.lerp((i+1) as f32 * binsf_inf);
            }

            // compute best combination; minimal cost
            let (mut ls, mut rs) = (0, 0);
            lb.set_default();
            rb.set_default();
            let max_cost = count as f32 * top_bound.surface_area();
            let mut best_cost = max_cost;

            for axis in [Axis::X, Axis::Y, Axis::Z] {

                let u = axis.as_usize();
                if !axis_valid[u] {
                    continue;
                }
                let k1 = (binsf*(1.0-EPSILON))/(top_bound.max.fake_arr(axis)-top_bound.min.fake_arr(axis));
                let k0 = top_bound.min.fake_arr(axis);

                // place bounds in bins
                // generate bounds of bins
                binbounds.fill(aabb_null);
                bincounts.fill(0);
                let mut index: usize ;
                for index_triangle in sub_range.clone() {
                    index = (k1*(bounds[index_triangle].midpoint().as_array()[u]-k0)) as usize;
                    binbounds[index].combine(bounds[index_triangle]);
                    bincounts[index] += 1;
                }

                // iterate over bins
                for (lerp_index,lerp) in lerps.iter().enumerate(){
                    let split = lerp.fake_arr(axis);
                    // reset values
                    ls = 0;
                    rs = 0;
                    lb.set_default();
                    rb.set_default();
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
                        // println!("{},{:?},{}", cost,axis,split);
                        best_cost = cost;
                        best_axis = axis;
                        best_split = split;
                        best_aabb_left = lb;
                        best_aabb_right = rb;
                    }
                }
            }

            if best_cost == max_cost { // leaf
                v.left_first = first; // first
                v.count = count;
                continue;
            }

            // partition
            let mut a = first; // first
            let mut b = first + count - 1; // last
            let u = best_axis.as_usize();
            while a <= b{
                if bounds[a].midpoint().as_array()[u] < best_split{
                    a += 1;
                } else {
                    bounds.swap(a, b);
                    triangles.swap(a, b);
                    b -= 1;
                }
            }
            let l_count = a - first;

            if l_count == 0 || l_count == count{ // leaf
                v.left_first = first; // first
                v.count = count;
                continue;
            }
            v.count = 0; // internal vertex, not a leaf
            v.left_first = poolptr; // left = poolptr, right = poolptr + 1
            poolptr += 2;
            let lf = v.left_first;

            stack.push(StackItem {current: lf,first,count: l_count, depth: depth + 1});
            stack.push(StackItem {current: lf+1,first: first+l_count,count: count-l_count, depth: depth + 1});
        }
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

    pub fn intersect(&self, ray: Ray, scene: &Scene, hit: &mut RayHit) -> (usize, usize){

        fn internal_intersect(bvh: &Bvh, current: usize, ray: Ray, scene: &Scene, hit: &mut RayHit, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> (usize, usize){
            let vs = &bvh.vertices;
            let v = vs[current];
            if v.count > 0{ // leaf
                for i in v.left_first as usize..v.left_first as usize + v.count as usize {
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

    pub fn from_mesh(mesh: Mesh, triangles: &mut Vec<Triangle>, bins: usize) -> Self{
        let n = triangles.len();
        if n == 0 {
            return Self{ vertices: vec![], mesh };
        }
        let mut vs = vec![Vertex::default(); n * 2];
        let mut bounds = (0..n).into_iter().map(|i|
            AABB::from_points(&[triangles[i].a, triangles[i].b, triangles[i].c])
        ).collect::<Vec<_>>();

        Self::subdivide(&mut bounds, &mut vs, bins, triangles);

        Self{
            vertices: vs,
            mesh
        }
    }
}


const STACK_SIZE : usize = 100000;
pub struct CustomStack<T> {
    pub stack: [T; STACK_SIZE],
    pub index: usize
}

impl<T: Default + Copy> CustomStack<T> {
    pub fn new() -> Self{
        Self { stack: [T::default(); STACK_SIZE], index: 0 }
    }
    #[inline]
    pub fn current(&self) -> &T{
        &self.stack[self.index]
    }
    pub fn push(&mut self, item: T){
        self.index += 1;
        self.stack[self.index] = item;
    }
    pub fn pop(&mut self) -> T{
        self.index = self.index-1;
        self.stack[self.index + 1]
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct StackItem {
    pub current : usize,
    pub first : usize,
    pub count : usize,
    pub depth : usize
}
fn union_bound(bounds: &[AABB]) -> AABB {
    let mut bound = AABB::default();
    for other in bounds{
        bound.combine(*other);
    }
    bound.grown(Vec3::EPSILON)
}

#[cfg(test)]
mod tests {
    extern crate test;
    use test::Bencher;
    use crate::scene::Triangle;
    use crate::vec3::Vec3;
    use crate::bvh::Bvh;
    use crate::mesh::Mesh;

    #[bench]
    fn bench_bvh(b: &mut Bencher) {

        // credit: George Marsaglia
        #[inline]
        fn xor32(seed: &mut u32) -> u32{
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            *seed
        }

        // generate 5.000.000 triangles randomly
        let mut triangles = vec![];
        let mut seed:u32 = 81349324; // guaranteed to be random

        for i in 0..5000000{
            if i % 100000 == 0 {
                println!("{}",i);
            }
            triangles.push(Triangle{
                a: Vec3 { x: xor32(&mut seed) as f32, y: xor32(&mut seed) as f32, z: xor32(&mut seed) as f32 },
                b: Vec3 { x: xor32(&mut seed) as f32, y: xor32(&mut seed) as f32, z: xor32(&mut seed) as f32 },
                c: Vec3 { x: xor32(&mut seed) as f32, y: xor32(&mut seed) as f32, z: xor32(&mut seed) as f32 },
            });
            // println!("{},{:?}",i, triangles[i]);
        }
        b.bench(|_| { Bvh::from_mesh(Mesh::default(), &mut triangles, 12); });
    }
}
