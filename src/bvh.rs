use crate::scene::{ Scene, MeshIndex, Triangle, Intersectable };
use crate::cpu::inter::*;
use crate::aabb::*;
use crate::vec3::Vec3;
use crate::consts::{ EPSILON };
use crate::primitive::Primitive;

pub enum ContainerType {
    MESH,
    TOP
}
impl Default for ContainerType {
    fn default() -> Self { ContainerType::MESH }
}

#[derive(Default)]
pub struct Bvh{
    pub vertices: Vec<Vertex>,
    pub mesh_index: MeshIndex,
    pub quality: u8,
    pub container_type: ContainerType
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex{
    pub bound: AABB,
    pub left_first: usize,
    pub count: usize,
}

impl Bvh{
    #[allow(clippy::too_many_arguments)]
    fn subdivide<Q>(bounds: &mut Vec<AABB>, vs: &mut Vec<Vertex>, items: &mut Vec<Q>, bins: usize, quality: u8){
        let current = 0;
        let first = 0;
        let mut poolptr = 2;
        let count = items.len();
        let mut depth = 0;

        let binsf = bins as f32;
        let binsf_inf = 1.0 / binsf;

        let mut stack = vec![StackItem{ current, first, count, depth }]; // [(current,first,count,step)]

        let mut lerps = vec![Vec3::ZERO; bins-1];
        let mut binbounds = vec![AABB::new();bins];
        let mut bincounts : Vec<usize> = vec![0;bins];
        let aabb_null = AABB::default();

        let (mut lb, mut rb) = (AABB::default(), AABB::default());

        let mut best_axis = Axis::X;
        let mut best_split = 0.0;

        let mut v;
        let mut top_bound;
        let mut sub_range;

        let mut current;
        let mut first;
        let mut count;

        let mut depth_timers = vec![];
        let mut depth_items = vec![];

        while !stack.is_empty() {
            let x = stack.pop().unwrap();
            // println!("{:?}", x);
            depth = x.depth;
            if depth >= depth_timers.len() {
                depth_timers.push(0);
                depth_items.push(0);
            }

            depth_items[depth] += x.count;
            current = x.current;
            count = x.count;
            first = x.first;
            v = &mut vs[current];
            sub_range = first..first + count;
            top_bound = union_bound(&bounds[sub_range.clone()]);
            v.bound = top_bound;

            // find split
            if quality == 1 { // sah binned
                if count < 3 { // leaf
                    v.left_first = first; // first
                    v.count = count;
                    continue;
                }
                let diff = top_bound.max.subed(top_bound.min);
                let axis_valid = [diff.x > binsf * EPSILON, diff.y > binsf * EPSILON, diff.z > binsf * EPSILON];

                // precompute lerps
                for (i, item) in lerps.iter_mut().enumerate(){
                    *item = top_bound.lerp((i+1) as f32 * binsf_inf);
                }

                // compute best combination; minimal cost
                let (mut ls, mut rs);
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
                        }
                    }
                }
                if best_cost == max_cost { // leaf
                    v.left_first = first; // first
                    v.count = count;
                    continue;
                }
            }
            else { // midpoint
                // todo: midpoint heuristics
                if count < 5 { // leaf
                    v.left_first = first; // first
                    v.count = count;
                    continue;
                }
                // dominant axis
                let diff_x = top_bound.max.x - top_bound.min.x;
                let diff_y = top_bound.max.y - top_bound.min.y;
                let diff_z = top_bound.max.z - top_bound.min.z;
                let axis = if diff_x > diff_y && diff_x > diff_z { Axis::X }
                else if diff_y > diff_z { Axis::Y }
                else { Axis::Z };
                // midpoint
                best_split = top_bound.midpoint().fake_arr(axis);
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
                    items.swap(a, b);
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

    pub fn from_primitives(bounds: &mut Vec<AABB>, primitives: &mut Vec<Primitive>) -> Self{
        let bins = 12;
        let quality = 1;
        let n = primitives.len();
        let mut vs = vec![Vertex::default(); n * 2];
        Self::subdivide::<Primitive>(bounds, &mut vs, primitives, bins, quality);
        // todo add primitives to gpu array buffer
        Self{
            vertices: vs,
            mesh_index: 0,
            quality,
            container_type: ContainerType::TOP
        }
    }

    pub fn from_mesh(mesh_index: MeshIndex, triangles: &mut Vec<Triangle>, bins: usize) -> Self{
        let quality = 1;
        let n = triangles.len();
        assert!(n > 0);
        let mut vs = vec![Vertex::default(); n * 2];
        let mut bounds = (0..n).into_iter().map(|i|
            AABB::from_points(&[triangles[i].a, triangles[i].b, triangles[i].c])
        ).collect::<Vec<_>>();
        Self::subdivide::<Triangle>(&mut bounds, &mut vs, triangles, bins, quality);

        Self{
            vertices: vs,
            mesh_index,
            quality,
            container_type: ContainerType::MESH
        }
    }

    pub fn get_item_count(&self, current: usize, vec: &mut Vec<usize>){
        if current >= self.vertices.len() { return; }
        let vs = &self.vertices;
        let v = vs[current];
        if v.count > 0{ // leaf
            vec.push(v.count);
        } else { // vertex
            self.get_item_count(v.left_first, vec);
            self.get_item_count(v.left_first + 1, vec);
        }
    }

    pub fn intersect(&self, ray: Ray, scene: &Scene, hit: &mut RayHit) -> (usize, usize){
        fn internal_intersect(bvh: &Bvh, current: usize, ray: Ray, scene: &Scene, hit: &mut RayHit, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> (usize, usize){
            let vs = &bvh.vertices;
            let v = vs[current];
            let mut a = 0; let mut b = 0;
            if v.count > 0{ // leaf
                match bvh.container_type {
                    ContainerType::MESH => { // triangle from mesh
                        let mesh = &scene.meshes[bvh.mesh_index as usize];
                        for i in (v.left_first) as usize..(v.left_first+v.count) as usize {
                            // intersect triangle
                            mesh.get_triangle(i, scene).intersect(ray, hit);
                        }
                    },
                    ContainerType::TOP => { // primitive from scene
                        for i in (v.left_first) as usize..(v.left_first+v.count) as usize {
                            // intersect primitive
                            let primitive = &scene.primitives[i];
                            let (_a, _b) = primitive.intersect(ray, scene, hit);
                            a += _a;
                            b += _b;
                        }
                    }
                }
                (a, v.count as usize + b)
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

                if ts[order[0]] >= 0.0 && ts[order[0]] < hit.t {
                    x1 = internal_intersect(bvh, nodes[order[0]], ray, scene, hit, inv_dir, dir_is_neg);
                    if ts[order[1]] >= 0.0 && ts[order[1]] < hit.t {
                        x2 = internal_intersect(bvh, nodes[order[1]], ray, scene, hit, inv_dir, dir_is_neg);
                    }
                } else if ts[order[1]] >= 0.0 && ts[order[1]] < hit.t {
                    x2 = internal_intersect(bvh, nodes[order[1]], ray, scene, hit, inv_dir, dir_is_neg);
                }
                (2 + x1.0 + x2.0, 1 + x1.1.max(x2.1))
            }
        }

        // TODO: doesn't work anymore with bvh? Problem in texture code
        // for plane in &scene.planes { inter_plane(ray, plane, hit); }
        assert!(!self.vertices.is_empty());
        let inv_dir = ray.inverted().dir;
        let dir_is_neg : [usize; 3] = ray.direction_negations();
        internal_intersect(self, 0, ray, scene, hit, inv_dir, dir_is_neg)
    }

    // this occluded function is a hard copy of intersect with trivial changes to return early.
    // any non-trivial changes should be made in intersect and be duplicated to this occluded function.
    pub fn occluded(&self, ray: Ray, scene: &Scene, dist: f32) -> bool{
        #[allow(clippy::too_many_arguments)]
        fn internal_intersect(bvh: &Bvh, current: usize, ray: Ray, scene: &Scene, hit: &mut RayHit, dist: f32, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> bool{
            let vs = &bvh.vertices;
            let v = vs[current];
            if v.count > 0{ // leaf
                match bvh.container_type {
                    ContainerType::MESH => { // triangle from mesh
                        let mesh = &scene.meshes[bvh.mesh_index as usize];
                        for i in (v.left_first) as usize..(v.left_first+v.count) as usize {
                            // intersect triangle
                            if dist_triangle(ray, &mesh.get_triangle(i, scene)) < dist { return true; }
                        }
                    },
                    ContainerType::TOP => { // primitive from scene
                        for i in (v.left_first) as usize..(v.left_first+v.count) as usize {
                            // intersect primitive
                            let primitive = &scene.primitives[i];
                            if primitive.occluded(ray, scene, dist) { return true; }
                        }
                    }
                }
                hit.t < dist
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

                if ts[order[0]] >= 0.0 && ts[order[0]] < dist {
                    if internal_intersect(bvh, nodes[order[0]], ray, scene, hit, dist, inv_dir, dir_is_neg) { return true; }
                    if ts[order[1]] >= 0.0 && ts[order[1]] < dist && internal_intersect(bvh, nodes[order[1]], ray, scene, hit, dist, inv_dir, dir_is_neg) { return true; }
                }
                else if ts[order[1]] >= 0.0 && ts[order[1]] < dist && internal_intersect(bvh, nodes[order[1]], ray, scene, hit, dist, inv_dir, dir_is_neg) { return true; }
                false
            }
        }

        // TODO: doesn't work anymore with bvh? Problem in texture code
        // for plane in &scene.planes { inter_plane(ray, plane, hit); }
        assert!(!self.vertices.is_empty());
        let mut hit = RayHit::NULL;
        let inv_dir = ray.inverted().dir;
        let dir_is_neg : [usize; 3] = ray.direction_negations();
        internal_intersect(self, 0, ray, scene, &mut hit, dist, inv_dir, dir_is_neg)
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
        b.bench(|_| { Bvh::from_mesh(0, &mut triangles, 12); });
    }
}
