use crate::scene::{Scene, MeshIndex, Triangle, Intersectable };
use crate::cpu::inter::*;
use crate::aabb::*;
use crate::vec3::Vec3;
use crate::consts::{ EPSILON };
use crate::primitive::{Primitive, Shape};

pub enum ContainerType {
    MESH,
    TOP
}
impl Default for ContainerType {
    fn default() -> Self { ContainerType::MESH }
}

fn assert_almost_equal(a: f32, b: f32) {
    if (a-b).abs() > EPSILON {
        assert_eq!(a,b);
    }
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
    fn subdivide<Q: Intersectable + Clone >(bounds: &mut Vec<AABB>, vs: &mut Vec<Vertex>, items: &mut Vec<Q>, bins: usize, quality: u8, is_toplevel: bool){
        // let alpha = 1.0;
        // let alpha = 0.001;
        let alpha = 0.00001;

        let before = items.len();
        let current = 0;
        let first = 0;
        let mut poolptr = 2;
        let count = items.len();
        let mut depth = 0;
        let mut inserted = 0; // number of items inserted

        let binsf = bins as f32;
        let binsf_inf = 1.0 / binsf;

        let mut stack = vec![StackItem{ current, first, count, depth, inserted}];

        let mut lerps = vec![Vec3::ZERO; bins-1];
        let mut binbounds = vec![AABB::new();bins];
        let mut bincounts : Vec<usize> = vec![0;bins];
        let mut startcounts : Vec<usize> = vec![0;bins]; // for spatial splits
        let mut endcounts : Vec<usize> = vec![0;bins];   // for spatial splits
        let aabb_null = AABB::default();

        let (mut lb, mut rb) = (AABB::default(), AABB::default());

        let mut best_axis = Axis::X;
        let mut best_split = 0.0;
        let mut best_type = 0;
        let mut best_lb = aabb_null;
        let mut best_rb = aabb_null;
        let mut best_object_split = 0.0;
        let mut best_spatial_split = 0.0;

        let mut v;
        let mut top_bound;
        let mut sub_range;

        let mut current;
        let mut first;
        let mut last; // use for convenience to pinpoint last item in range for current stack item
        let mut count;

        let mut root_sa= f32::MAX;

        while !stack.is_empty() {
            let x = stack.pop().unwrap();
            // println!("{:?}", x);
            depth = x.depth;
            current = x.current;
            count = x.count;
            first = x.first + inserted - x.inserted; // account for inserted elements done after getting pushed to the stack
            last  = first + x.count;

            v = &mut vs[current];
            sub_range = first..first + count;
            top_bound = union_bound(&bounds[sub_range.clone()]);
            if depth == 0 { root_sa = top_bound.surface_area(); }
            v.bound = top_bound;

            // find split
            if quality == 1 { // sah binned
                if count < 3 { // leaf
                    v.left_first = first; // first
                    v.count = count;
                    continue;
                }
                let diff = top_bound.max.subed(top_bound.min);
                let axis_valid = [
                    diff.x > binsf * EPSILON,
                    diff.y > binsf * EPSILON,
                    diff.z > binsf * EPSILON
                ];

                // precompute lerps
                for (i, item) in lerps.iter_mut().enumerate(){
                    *item = top_bound.lerp((i+1) as f32 * binsf_inf);
                }

                // compute best combination/minimal cost for object split
                let (mut ls, mut rs);
                lb.set_default();
                rb.set_default();
                let max_cost = count as f32 * top_bound.surface_area();
                let mut best_overlap = AABB::new();
                let mut best_cost = max_cost;

                // find best object split
                for axis in [Axis::X, Axis::Y, Axis::Z] {

                    let u = axis.as_usize();
                    if !axis_valid[u] {
                        continue;
                    }
                    let k1 = (binsf*(1.0-EPSILON))/(top_bound.max.fake_arr(axis)-top_bound.min.fake_arr(axis));
                    let k0 = top_bound.min.fake_arr(axis);

                    // bounds of bins
                    binbounds.fill(aabb_null);
                    bincounts.fill(0);
                    let mut index: usize ;
                    for index_item in sub_range.clone() {
                        index = (k1*(bounds[index_item].midpoint().as_array()[u]-k0)) as usize;
                        binbounds[index].combine(bounds[index_item]);
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
                            best_lb = lb;
                            best_rb = rb;
                            best_overlap = lb.overlap(rb);
                            best_cost = cost;
                            best_axis = axis;
                            best_split = split;
                            best_type = 0;
                            best_object_split = cost;
                        }
                    }
                }
                // compute lambda
                let lambda = best_overlap.surface_area() / root_sa;

                // find best spatial split
                if lambda > alpha {
                    // println!("{}/{} > {}", lambda, root_sa, alpha);
                    for axis in [Axis::X, Axis::Y, Axis::Z] {
                        // skip invalid axii
                        let u = axis.as_usize();
                        if !axis_valid[u] {
                            continue;
                        }
                        // bin-selection variables
                        let k1 = (binsf*(1.0-EPSILON))/(top_bound.max.fake_arr(axis)-top_bound.min.fake_arr(axis));
                        let k0 = top_bound.min.fake_arr(axis);

                        // bounds of bins
                        // fill in info for bins (bounds, counts, start, end)
                        binbounds.fill(aabb_null);
                        bincounts.fill(0);
                        startcounts.fill(0);
                        endcounts.fill(0);

                        // construct aabb of every lerp bin
                        let mut lerp_aabbs = vec![top_bound; bins];
                        for (lerp_index,lerp) in lerps.iter().enumerate() {
                            lerp_aabbs[lerp_index].split_by(axis, lerp.fake_arr(axis), 1);
                            lerp_aabbs[lerp_index+1].split_by(axis, lerp.fake_arr(axis), 0); // account for left side next item
                        }

                        for index_item in sub_range.clone() { // spatial
                            let bound = bounds[index_item];

                            // start count
                            let v = bound.min.fake_arr(axis);
                            let index_start = (k1*(v-k0)) as usize;
                            startcounts[index_start] += 1;

                            // end count
                            let v = bound.max.fake_arr(axis);
                            let index_end = (k1*(v-k0)).min(bins as f32 - 1.0) as usize;
                            endcounts[index_end] += 1;

                            // per bin, construct aabb of triangle clipped
                            let item: &Q = &items[index_item];
                            let bound = &bounds[index_item];
                            let mut tmpbound = AABB::new();
                            for i in (0..bins) {
                                let subbound = item.clip(lerp_aabbs[i], bound);
                                binbounds[i].combine(subbound);
                            }
                        }
                        // iterate over bins
                        for i in 1..bins {
                            // reset values
                            ls = 0; // items in left
                            rs = 0; // items in right
                            lb.set_default(); // bounds of left
                            rb.set_default(); // bounds of right
                            let mut active = 0;
                            for j in 0..i { // left of split
                                ls += startcounts[j];
                                active += startcounts[j];
                                active -= endcounts[j];
                                lb.combine(binbounds[j]);
                            }
                            rs += active;
                            for j in i..bins { // right of split
                                rs += startcounts[j];
                                active += startcounts[j];
                                active -= endcounts[j];
                                rb.combine(binbounds[j]);
                            }
                            let cost = 3.0 + 1.0 + lb.surface_area() * ls as f32 + 1.0 + rb.surface_area() * rs as f32;
                            if cost < best_cost {
                                best_overlap = lb.overlap(rb);
                                best_cost = cost;
                                best_axis = axis;
                                best_type = 1;
                                best_lb   = lb;
                                best_rb   = rb;
                                best_spatial_split = cost;
                            }
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
                best_axis = axis;
                best_split = top_bound.midpoint().fake_arr(axis);
                best_type = 0;
            }

            // partition
            let mut a = first; // first
            let mut b = first + count - 1; // last
            if best_type == 1 { // spatial split
                while a <= b {
                    let bound_left = items[a].clip(best_lb, &bounds[a]);
                    let bound_right = items[a].clip(best_rb, &bounds[a]);
                    if bound_left.surface_area() > 0.0 {
                        bounds[a] = bound_left;
                        if bound_right.surface_area() > 0.0 {
                            bounds.insert(last, bound_right);
                            items.insert(last, items[a].clone());
                            count += 1; // one additional item to handle in right child
                            inserted += 1;
                        }
                        a += 1;
                    } else if bound_left.surface_area() == 0.0 && bound_right.surface_area() > 0.0 {
                        bounds[a] = bound_right;
                        bounds.swap(a, b);
                        items.swap(a, b);
                        if b == 0 { a += 1; }
                        else { b -= 1; }
                    } else {
                        panic!("both left and right are empty");
                    }
                }
            } else { // object split
                // swap in such that items in appropriate bin interval based on midpoint
                while a <= b{
                    if bounds[a].midpoint().fake_arr(best_axis) < best_split{
                        a += 1;
                    } else {
                        bounds.swap(a, b);
                        items.swap(a, b);
                        if b == 0 { a += 1; }
                        else { b -= 1; }
                    }
                }
            }
            let l_count = a - first;

            // check all items on one side of split plane
            if l_count == 0 || l_count == count{ // leaf
                v.left_first = first; // first
                v.count = count;
                continue;
            }
            v.count = 0; // internal vertex, not a leaf
            v.left_first = poolptr; // left = poolptr, right = poolptr + 1
            poolptr += 2;
            let lf = v.left_first;
            // evaluate children
            stack.push(StackItem {current: lf+1,first: first+l_count,count: count-l_count, depth: depth + 1, inserted});
            stack.push(StackItem {current: lf  ,first               ,count: l_count      , depth: depth + 1, inserted});
        }

        let after = items.len();
        println!("before ({}), after({})", before, after);
    }

    pub fn from_primitives(bounds: &mut Vec<AABB>, primitives: &mut Vec<Primitive>) -> Self{
        let bins = 12;
        let quality = 1;
        let n = primitives.len();
        let mut vs = vec![Vertex::default(); n * 2];
        Self::subdivide::<Primitive>(bounds, &mut vs, primitives, bins, quality, true);
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
        Self::subdivide::<Triangle>(&mut bounds, &mut vs, triangles, bins, quality, false);

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

                if ts[order[0]] >= 0.0 &&
                    ts[order[0]] < dist {
                    if internal_intersect(bvh, nodes[order[0]], ray, scene, hit, dist, inv_dir, dir_is_neg) {
                        return true;
                    }
                    if ts[order[1]] >= 0.0 &&
                        ts[order[1]] < dist &&
                        internal_intersect(bvh, nodes[order[1]], ray, scene, hit, dist, inv_dir, dir_is_neg) {
                        return true;
                    }
                }
                else if ts[order[1]] >= 0.0 &&
                        ts[order[1]] < dist &&
                        internal_intersect(bvh, nodes[order[1]], ray, scene, hit, dist, inv_dir, dir_is_neg) {
                    return true;
                }
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
    pub depth : usize,
    pub inserted: usize,
}

fn union_bound(bounds: &[AABB]) -> AABB {
    let mut bound = AABB::default();
    for other in bounds{
        bound.combine(*other);
    }
    bound.grown(Vec3::EPSILON)
}
