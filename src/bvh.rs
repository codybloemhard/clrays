use crate::vec3::Vec3;
use crate::consts::{EPSILON, UV_SPHERE, UV_PLANE, MAX_RENDER_DIST};
use crate::scene::{Sphere, Plane, Triangle, Scene};
use crate::cpu::{Ray, RayHit};

#[inline]
fn gamma(n: i32) -> f32 {
    (n as f32 * EPSILON) / (1.0 - n as f32 * EPSILON)
}

// ray-sphere intersection
#[inline]
pub fn inter_sphere<'a>(ray: Ray, sphere: &'a Sphere, closest: &mut RayHit<'a>){
    let l = Vec3::subed(sphere.pos, ray.pos);
    let tca = Vec3::dot(ray.dir, l);
    let d = tca*tca - Vec3::dot(l, l) + sphere.rad*sphere.rad;
    if d < 0.0 { return; }
    let dsqrt = d.sqrt();
    let mut t = tca - dsqrt;
    if t < 0.0 {
        t = tca + dsqrt;
        if t < 0.0 { return; }
    }
    if t > closest.t { return; }
    closest.t = t;
    closest.pos = ray.pos.added(ray.dir.scaled(t));
    closest.nor = Vec3::subed(closest.pos, sphere.pos).scaled(1.0 / sphere.rad);
    closest.mat = Some(&sphere.mat);
    closest.uvtype = UV_SPHERE;
    closest.sphere = Some(sphere);
}


// ray-plane intersection
#[inline]
pub fn inter_plane<'a>(ray: Ray, plane: &'a Plane, closest: &mut RayHit<'a>){
    let divisor = Vec3::dot(ray.dir, plane.nor);
    if divisor.abs() < EPSILON { return; }
    let planevec = Vec3::subed(plane.pos, ray.pos);
    let t = Vec3::dot(planevec, plane.nor) / divisor;
    if t < EPSILON { return; }
    if t > closest.t { return; }
    closest.t = t;
    closest.pos = ray.pos.added(ray.dir.scaled(t));
    closest.nor = plane.nor;
    closest.mat = Some(&plane.mat);
    closest.uvtype = UV_PLANE;
    closest.sphere = None;
}

// ray-triangle intersection
#[inline]
#[allow(clippy::many_single_char_names)]
pub fn inter_triangle<'a>(ray: Ray, tri: &'a Triangle, closest: &mut RayHit<'a>){
    let edge1 = Vec3::subed(tri.b, tri.a);
    let edge2 = Vec3::subed(tri.c, tri.a);
    let h = Vec3::crossed(ray.dir, edge2);
    let a = Vec3::dot(edge1, h);
    if a > -EPSILON*0.01 && a < EPSILON*0.01 { return; } // ray parallel to tri
    let f = 1.0 / a;
    let s = Vec3::subed(ray.pos, tri.a);
    let u = f * Vec3::dot(s, h);
    if !(0.0..=1.0).contains(&u) { return; }
    let q = Vec3::crossed(s, edge1);
    let v = f * Vec3::dot(ray.dir, q);
    if v < 0.0 || u + v > 1.0 { return; }
    let t = f * Vec3::dot(edge2, q);
    if t <= EPSILON { return; }
    if t > closest.t { return; }
    closest.t = t;
    closest.pos = ray.pos.added(ray.dir.scaled(t));
    closest.nor = Vec3::crossed(edge1, edge2).normalized_fast();
    closest.mat = Some(&tri.mat);
    closest.uvtype = UV_PLANE;
    closest.sphere = None;
}

#[derive(Copy, Clone, Debug)]
pub struct AABB { // 24 bytes
    pub a: Vec3, // Vec3: 12 bytes
    pub b: Vec3, // Vec3: 12 bytes
}
impl AABB {
    pub fn new() -> Self{
        Self {
            a: Vec3 { x: f32::MAX, y: f32::MAX, z: f32::MAX, },
            b: Vec3 { x: f32::MIN, y: f32::MIN, z: f32::MIN, },
        }
    }

    pub fn midpoint(&self) -> Vec3{
        self.lerp(0.5)
    }

    pub fn lerp(&self, val: f32) -> Vec3{
        self.a.scaled(1.0-val).added(
            self.b.scaled(val)
        )
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

#[derive(Copy, Clone, Debug)]
enum Shape { // ? bytes
    NONE,
    TRIANGLE,
    SPHERE,
    PLANE
}

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

#[derive(Clone, Copy)]
pub struct Primitive {
    bounds: AABB,
    shape_type: Shape,
    sphere: usize,
    plane: usize,
    triangle: usize,
}
impl Primitive {

    fn from_sphere(sphere: &Sphere, index_sphere: usize) -> Self{
        Self {
            bounds: AABB {
                a: Vec3 {
                    x: sphere.pos.x - sphere.rad,
                    y: sphere.pos.y - sphere.rad,
                    z: sphere.pos.z - sphere.rad,
                },
                b : Vec3 {
                    x: sphere.pos.x + sphere.rad,
                    y: sphere.pos.y + sphere.rad,
                    z: sphere.pos.z + sphere.rad,
                }
            },
            shape_type: Shape::SPHERE,
            sphere: index_sphere,
            plane: 0,
            triangle: 0,
        }
    }

    fn from_plane(plane: &Plane, index_plane: usize) -> Self{
        if !(plane.nor.x == 1.0 || plane.nor.y == 1.0 || plane.nor.z == 1.0) {
            panic!("Infinite plane with unaligned normal: Bounding box will be infinite");
        }
        Self {
            bounds: AABB {
                a: Vec3 {
                    x: -f32::MAX, // Using f32::INFINITY crashes my computer
                    y: plane.pos.y,
                    z: -f32::MAX // Using f32::INFINITY crashes my computer
                },
                b: Vec3 {
                    x: f32::MAX, // Using f32::INFINITY crashes my computer
                    y: plane.pos.y + EPSILON,
                    z: f32::MAX, // Using f32::INFINITY crashes my computer
                }
            },
            shape_type: Shape::PLANE,
            sphere: 0,
            plane: index_plane,
            triangle: 0
        }
    }

    fn from_triangle(triangle: &Triangle, index_triangle: usize) -> Self{
        Self {
            bounds: AABB {
                a: Vec3 {
                    x: triangle.a.x.min(triangle.b.x).min(triangle.c.x),
                    y: triangle.a.y.min(triangle.b.y).min(triangle.c.y),
                    z: triangle.a.z.min(triangle.b.z).min(triangle.c.z),
                }.subed(Vec3::EPSILON),
                b : Vec3 {
                    x: triangle.a.x.max(triangle.b.x).max(triangle.c.x),
                    y: triangle.a.y.max(triangle.b.y).max(triangle.c.y),
                    z: triangle.a.z.max(triangle.b.z).max(triangle.c.z),
                }.added(Vec3::EPSILON)
            },
            shape_type: Shape::TRIANGLE,
            sphere: 0,
            plane: 0,
            triangle: index_triangle,
        }
    }

    pub fn intersect<'a>(&self, ray: Ray, scene: &'a Scene, closest: &mut RayHit<'a>) {
        match self.shape_type {
            Shape::SPHERE => inter_sphere(ray, &scene.spheres[self.sphere], closest),
            Shape::PLANE => inter_plane(ray, &scene.planes[self.plane], closest),
            Shape::TRIANGLE => inter_triangle(ray, &scene.triangles[self.triangle], closest),
            Shape::NONE => {}
        }
    }
}

// #[derive(Copy, Clone, Debug)]
pub struct Node {
    pub bounds: AABB,  // AABB: 24 bytes
    pub is_leaf: bool, // bool: 1 bit
    pub primitives: Vec<Primitive>,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,
}

impl Node {
    pub fn build_sah(primitives: &Vec<Primitive>, depth: usize) -> Self {
        // apply SAH
        // make 12 splits

        // find bounds
        let mut bounds = AABB::new();
        for primitive in primitives.iter() {
            bounds.a.x = bounds.a.x.min( primitive.bounds.a.x );
            bounds.a.y = bounds.a.y.min( primitive.bounds.a.y );
            bounds.a.z = bounds.a.z.min( primitive.bounds.a.z );

            bounds.b.x = bounds.b.x.max( primitive.bounds.b.x );
            bounds.b.y = bounds.b.y.max( primitive.bounds.b.y );
            bounds.b.z = bounds.b.z.max( primitive.bounds.b.z );
        }

        let diff = bounds.b.subed(bounds.a);
        let axis_valid = [diff.x > 12.0*EPSILON, diff.y > 12.0*EPSILON, diff.z > 12.0*EPSILON ];

        // precompute lerps
        let mut lerps = [Vec3::DOWN; 12];
        for i in 0..12{
            lerps[i] = bounds.lerp(i as f32 / 12.0 );
        }

        // precompute midpoints
        let mid_points_prims: Vec<Vec3> = primitives.into_iter().map(|prim| prim.bounds.midpoint()).collect();
        let n = mid_points_prims.len();

        // compute best combination; minimal cost
        let mut best : (f32, Axis, usize) = (f32::MAX, Axis::X, 0); // (cost, axis, i_lerp)
        for axis in [Axis::X, Axis::Y, Axis::Z] {

            if !axis_valid[axis.as_usize()] {
                // println!("Axis {:?} invalid", axis);
                continue;
            }

            let mut index_reshuffled = vec![0; n];
            for i in 0..n{
                index_reshuffled[i] = i;
            }

            // TODO: improve algorithm to O(n*log(n))
            // sort primitives for axis
            let vals : Vec<f32> = mid_points_prims.iter().map(|point| point.fake_arr(axis)).collect();
            for i in 0..n{
                let val = vals[i];
                let mut x = 0;
                for j in 0..n{
                    if vals[j] < val {
                        x += 1;
                    } else if vals[j] == val {
                        if j < i {
                            x += 1;
                        }
                    }
                }
                index_reshuffled[x] = i;
            }

            // temporarily reshuffling is ok
            // println!("{:?}",vals);
            for a in 0..n{
                for b in a+1..n{
                    assert!(index_reshuffled[a] != index_reshuffled[b]);
                    // println!("{},{}",vals[index_reshuffled[a]],vals[index_reshuffled[b]]);
                    assert!(vals[index_reshuffled[a]] <= vals[index_reshuffled[b]]);
                }
            }

            let mut sides: [Vec<&Primitive>;2] = [vec![],vec![]];
            let mut side_bounds : [AABB; 2] = [AABB::new(); 2];
            for i in 0..n{
                // initially all prims on the right
                // push them in reverse, so we can then easily pop later on
                sides[1].push(&primitives[index_reshuffled[n-i-1]]);
            }
            let mut i_split: usize = 0;

            for i_lerp in 0..11{
                // println!("{:?}", i_lerp);
                // println!("{:?}", axis.as_usize());
                // println!("{:?}", lerps);
                // println!("{:?}>{:?}", lerps[i_lerp+1].fake_arr(axis), lerps[i_lerp].fake_arr(axis));
                assert!(lerps[i_lerp+1].fake_arr(axis) > lerps[i_lerp].fake_arr(axis));
            }

            // iterate over 12 bins
            for i_lerp in 0..12{
                let tmp = lerps[i_lerp].fake_arr(axis);

                // place over prims from right split to left split
                while i_split < n && mid_points_prims[index_reshuffled[i_split]].fake_arr(axis) < tmp{
                    let prim = sides[1].pop().unwrap();
                    assert_eq!(mid_points_prims[index_reshuffled[i_split]].fake_arr(axis), prim.bounds.midpoint().fake_arr(axis) );
                    assert!(prim.bounds.midpoint().fake_arr(axis) < tmp );
                    sides[0].push(prim);
                    assert!(sides[0][i_split].bounds.midpoint().fake_arr(axis) < tmp);
                    i_split += 1;
                }

                // check all in left are smaller
                for a in 0..sides[0].len() {
                    for b in 0..sides[1].len() {
                        assert!(sides[0][a].bounds.midpoint().fake_arr(axis) < sides[1][b].bounds.midpoint().fake_arr(axis));
                    }
                }

                // check left is smaller than tmp
                for a in 0..sides[0].len() {
                    // println!("{} < {}?", sides[0][a].bounds.midpoint().fake_arr(axis), tmp);
                    assert!(sides[0][a].bounds.midpoint().fake_arr(axis) < tmp);
                }

                // check right is larger than tmp
                for b in 0..sides[1].len() {
                    assert!(sides[1][b].bounds.midpoint().fake_arr(axis) >= tmp);
                }

                // update bounds of left
                for prim in sides[0].iter() {
                    side_bounds[0].a.x = side_bounds[0].a.x.min(prim.bounds.a.x);
                    side_bounds[0].a.y = side_bounds[0].a.y.min(prim.bounds.a.y);
                    side_bounds[0].a.z = side_bounds[0].a.z.min(prim.bounds.a.z);

                    side_bounds[0].b.x = side_bounds[0].b.x.max(prim.bounds.b.x);
                    side_bounds[0].b.y = side_bounds[0].b.y.max(prim.bounds.b.y);
                    side_bounds[0].b.z = side_bounds[0].b.z.max(prim.bounds.b.z);
                }

                // recompute bounds for right
                for prim in sides[1].iter() {
                    side_bounds[1].a.x = side_bounds[1].a.x.min( prim.bounds.a.x );
                    side_bounds[1].a.y = side_bounds[1].a.y.min( prim.bounds.a.y );
                    side_bounds[1].a.z = side_bounds[1].a.z.min( prim.bounds.a.z );

                    side_bounds[1].b.x = side_bounds[1].b.x.max( prim.bounds.b.x );
                    side_bounds[1].b.y = side_bounds[1].b.y.max( prim.bounds.b.y );
                    side_bounds[1].b.z = side_bounds[1].b.z.max( prim.bounds.b.z );
                }

                // get cost
                let cost = side_bounds[0].surface_area()*sides[0].len() as f32 + side_bounds[1].surface_area()*sides[1].len() as f32;
                if cost < best.0 {
                    best = (cost, axis, i_lerp);
                }
            }
        }

        // apply best
        let (cost, axis, i_lerp) = best;
        let val = lerps[i_lerp].fake_arr(axis);

        // Define primitives for left and right
        let mut left  : Vec<Primitive> = vec![];
        let mut right : Vec<Primitive> = vec![];
        for i in 0..n {
            if mid_points_prims[i].fake_arr(axis) < val {
                left.push(primitives[i]);
            } else {
                right.push(primitives[i]);
            }
        }

        // TODO: improve cost function
        // decide whether we apply the primitives into a leaf node
        if left.len() == 0 || right.len() == 0 {
            // no cost gain
            let node = Node {
                bounds,
                primitives: primitives.clone(),
                is_leaf: true,
                left: None,
                right: None
            };
            return node;
        }

        // println!("depth: {}", depth);
        // println!("len left: {}", left.len());
        // println!("len right: {}", right.len());

        // Build left subnode and right subnode
        let node = Node {
            bounds,
            primitives: primitives.clone(),
            is_leaf: false,
            left: Some(Box::new(Node::build_sah(&left, depth + 1))),
            right: Some(Box::new(Node::build_sah(&right, depth + 1)))
        };
        return node;
    }

    pub fn build_mid(primitives: &Vec<Primitive>, depth: usize) -> Self {
        // use midpoint bvh

        // find bounds
        let mut bounds = AABB::new();
        for primitive in primitives.iter() {
            bounds.a.x = bounds.a.x.min( primitive.bounds.a.x );
            bounds.a.y = bounds.a.y.min( primitive.bounds.a.y );
            bounds.a.z = bounds.a.z.min( primitive.bounds.a.z );

            bounds.b.x = bounds.b.x.max( primitive.bounds.b.x );
            bounds.b.y = bounds.b.y.max( primitive.bounds.b.y );
            bounds.b.z = bounds.b.z.max( primitive.bounds.b.z );
        }

        let diff = bounds.b.subed(bounds.a);

        let axis_valid = [diff.x > 3.0*EPSILON, diff.y > 3.0*EPSILON, diff.z > 3.0*EPSILON ];

        // Find dominant axis
        let axis = if diff.x > diff.y && diff.x > diff.z && axis_valid[0] {
            Axis::X
        } else if diff.y > diff.z && axis_valid[1] {
            Axis::Y
        } else if axis_valid[2] {
            Axis::Z
        } else {
            panic!("No valid axis found");
        };

        // Find midpoint
        let mid_points_prims = primitives.into_iter().map(|prim| prim.bounds.midpoint()).collect::<Vec<Vec3>>();
        let vals = (&mid_points_prims).into_iter().map(|prim| prim.fake_arr(axis)).collect::<Vec<f32>>();
        let n = mid_points_prims.len();

        let mut val: f32 = 0.0;
        let mut first : f32 = f32::MIN;
        let mut is_all_the_same = true;
        for i in 0..n {
            let tmp = vals[i];
            if first != f32::MIN && (first - tmp).abs() > EPSILON {
                is_all_the_same = false;
            }
            first = tmp;
            val += tmp;
        }
        val /= primitives.len() as f32;

        // Decide whether we apply the primitives into a leaf node
        if primitives.len() < 5 || is_all_the_same {
            let node = Node {
                bounds,
                primitives: primitives.clone(),
                is_leaf: true,
                left: None,
                right: None
            };
            return node;
        }

        // Define primitives for left and right
        let mut left  : Vec<Primitive> = vec![];
        let mut right : Vec<Primitive> = vec![];
        for (i,primitive) in primitives.iter().enumerate() {
            let tmp = vals[i];
            if tmp < val {
                left.push(*primitive);
            } else {
                right.push(*primitive);
            }
        }

        // Build left subnode and right subnode
        let node = Node {
            bounds,
            primitives: primitives.clone(),
            is_leaf: false,
            left: Some(Box::new(Node::build_mid(&left, depth + 1))),
            right: Some(Box::new(Node::build_mid(&right, depth + 1)))
        };
        return node;
    }

    pub fn node_iterator(&self, call_on_every_node: &dyn Fn(&Node)) {
        call_on_every_node(&self);
        if !self.is_leaf {
            let node_left = self.left.as_deref().unwrap();
            node_left.node_iterator(call_on_every_node);
            let node_right = self.right.as_deref().unwrap();
            node_right.node_iterator(call_on_every_node);
        }
    }

    pub fn node_iterator_mut(&self, call_on_every_node: &mut dyn FnMut(&Node)) {
        call_on_every_node(&self);
        if !self.is_leaf {
            let node_left = self.left.as_deref().unwrap();
            node_left.node_iterator_mut(call_on_every_node);
            let node_right = self.right.as_deref().unwrap();
            node_right.node_iterator_mut(call_on_every_node);
        }
    }

    pub fn intersect<'a>(&self, ray: Ray, scene: &'a Scene, closest: &mut RayHit<'a>, inv_dir: Vec3, dir_is_neg: [usize; 3] ) -> (usize, usize, usize) {
        if self.is_leaf {
            for primitive in self.primitives.iter() {
                primitive.intersect(ray, scene, closest);
            }
            (0, self.primitives.len(), 1) // (aabb hits, prim hits, depth)
        } else {
            let nodes = [
                self.left.as_ref().unwrap(),
                self.right.as_ref().unwrap()
            ];

            let ts = [
                nodes[0].bounds.intersection(ray, inv_dir, dir_is_neg),
                nodes[1].bounds.intersection(ray, inv_dir, dir_is_neg)
            ];

            let order = if ts[0] <= ts[1] { [0,1] } else { [1,0] };
            let mut x1: (usize,usize,usize) = (0,0,0);
            let mut x2: (usize,usize,usize) = (0,0,0);
            if ts[order[0]] < closest.t {
                x1 = nodes[order[0]].intersect(ray, scene, closest, inv_dir, dir_is_neg);
                if ts[order[1]] < closest.t {
                    x2 = nodes[order[1]].intersect(ray, scene, closest, inv_dir, dir_is_neg);
                }
            }
            (2 + x1.0 + x2.0, x1.1 + x2.1, 1 + x1.2.max(x2.2))
        }
    }

    pub fn print(&self, depth: usize) {
        println!("{}", format!("{:>width$}", self.get_primitives_count(), width = 2*depth));
        if !self.is_leaf {
            self.left.as_ref().unwrap().print(depth + 1);
            self.right.as_ref().unwrap().print(depth + 1);
        }
    }

    pub fn get_primitives_count(&self) -> usize {
        if self.is_leaf {
            self.primitives.len()
        } else {
            self.left.as_ref().unwrap().get_primitives_count() + self.right.as_ref().unwrap().get_primitives_count()
        }
    }

    pub fn print_as_tree(&self) {
        let depth : usize = self.get_max_depth();
        let mut strings : Vec<String> = vec!["".parse().unwrap(); depth];
        self.print_get_subtree(&mut strings, 0);
        for string in strings.iter() {
            println!("{}", format!("{:^width$}", string, width = 400));
        }
    }

    pub fn print_get_subtree(&self, strings: &mut Vec<String>, depth: usize){
        if self.is_leaf {
            strings[depth] += &*format!("{:^width$}", self.get_primitives_count(), width = 5);
        } else {
            strings[depth] += &*format!("{:^width$}", self.get_primitives_count(), width = 5);
            let node_left = self.left.as_ref().unwrap();
            let node_right = self.left.as_ref().unwrap();

            node_left.print_get_subtree(strings, depth + 1);
            node_right.print_get_subtree(strings, depth + 1);
        }
    }

    pub fn get_max_depth(&self) -> usize {
        if self.is_leaf {
            1
        } else {
            1 + self.left.as_ref().unwrap().get_max_depth().max(self.right.as_ref().unwrap().get_max_depth())
        }
    }

}

pub struct BVH {
    pub node: Node,
}

impl BVH {
    pub fn build(scene: &Scene, is_sah: bool) -> BVH {
        let mut primitives: Vec<Primitive> = vec![];

        // Build primitives
        for i in 0..scene.spheres.len() {
            primitives.push(Primitive::from_sphere(&scene.spheres[i], i));
        };
        for i in 0..scene.planes.len() {
            primitives.push(Primitive::from_plane(&scene.planes[i], i));
        };
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };

        let root_node = if is_sah { Node::build_sah(&primitives, 0) } else { Node::build_mid(&primitives, 0) };
        BVH { node: root_node }
    }

    pub fn intersect<'a>(&self, ray: Ray, scene: &'a Scene, closest: &mut RayHit<'a>) -> (usize, usize, usize) {
        let inv_dir = ray.inverted().dir;
        let dir_is_neg : [usize; 3] = ray.direction_negations();
        self.node.intersect(ray, scene, closest, inv_dir, dir_is_neg)
    }
}


#[cfg(test)]
mod test {
    use crate::cpu::{Ray, RayHit, inter_scene};
    use crate::vec3::Vec3;
    use crate::bvh::{AABB, Primitive, Node};
    use crate::consts::{EPSILON, MAX_RENDER_DIST};
    use crate::scene::{Triangle, Material, Scene};
    use crate::mesh::{load_model, build_triangle_wall};
    use rand::random;

    #[test]
    fn unaligned_ray_bounding_boxes() {
        // In between
        let ray = Ray {
            pos: Vec3::ZERO,
            dir: Vec3::ONE.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_eq!(t, 0.0);

        // Before
        let ray = Ray {
            pos: Vec3::ONE.neged().scaled(5.0),
            dir: Vec3::ONE.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert!(t > 5.0 );

        // After
        let ray = Ray {
            pos: Vec3::ONE.scaled(5.0),
            dir: Vec3::ONE.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_eq!(t, MAX_RENDER_DIST);
    }

    #[inline]
    fn assert_small(a:f32,b:f32) {
        if (a-b).abs() > EPSILON { panic!("{} != {}", a,b); }
    }

    #[test]
    fn aligned_ray_bounding_boxes() {
        // In between
        let ray = Ray {
            pos: Vec3::ZERO,
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        // let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_small(t,  0.0);

        // Before
        let ray = Ray {
            pos: Vec3::FORWARD.neged().scaled(5.0),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_small(t, 4.0);

        // After
        let ray = Ray {
            pos: Vec3::FORWARD.scaled(5.0),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_small(t, MAX_RENDER_DIST);

    }

    #[test]
    fn aligned_ray_bounding_boxes_miss() {
        // In between
        let ray = Ray {
            pos: Vec3::ZERO.added(Vec3::UP.scaled(1.2) ),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_eq!(intersection, MAX_RENDER_DIST);

        // Before
        let ray = Ray {
            pos: Vec3::FORWARD.neged().scaled(5.0).added(Vec3::UP.scaled(1.2) ),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_eq!(intersection, MAX_RENDER_DIST);

        // After
        let ray = Ray {
            pos: Vec3::FORWARD.scaled(5.0).added(Vec3::UP.scaled(1.2) ),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_eq!(intersection, MAX_RENDER_DIST);
    }

    #[test]
    fn aligned_ray_bounding_boxes_miss_exact_same_position() {
        // In between
        let ray = Ray {
            pos: Vec3::ZERO.added(Vec3::UP ),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_small(t, 0.0);

        // Before
        let ray = Ray {
            pos: Vec3::FORWARD.neged().scaled(5.0).added(Vec3::UP ),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_small(t, 4.0);

        // After
        let ray = Ray {
            pos: Vec3::FORWARD.scaled(5.0).added(Vec3::UP ),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let t = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert_small(t, MAX_RENDER_DIST);
    }

    // #[test]
    // fn bounding_box_triangle() {
    //     let triangle = Triangle{
    //         a: Vec3{ x: 0.0, y: 0.0, z: 0.0 },
    //         b: Vec3{ x: 1.0, y: 0.0, z: 0.0 },
    //         c: Vec3{ x: 0.0, y: 1.0, z: 0.0 },
    //         mat: Material::basic()
    //     };
    //     let prim = Primitive::from_triangle(&triangle, 0);
    //     assert!(prim.bounds.a.equal(&triangle.a));
    //     assert!(prim.bounds.b.equal(&triangle.b.added(triangle.c)));
    // }

    #[test]
    fn ensure_aabb_has_volume(){
        let mut scene = Scene::new();
        // load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        build_triangle_wall(Material::basic(), &mut scene, 0.5, 10.0);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };
        let root_node_mid = Node::build_mid(&primitives, 0);
        let root_node_sah = Node::build_sah(&primitives, 0);
        for root_node in &[root_node_mid, root_node_sah] {
            root_node.node_iterator(&|s| assert!(s.bounds.volume() > 0.0));
        }
    }

    #[test]
    fn test_all_primitives_bound_within_node() {
        let mut scene = Scene::new();
        // load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        build_triangle_wall(Material::basic(), &mut scene, 0.5, 10.0);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };

        let root_node_mid = Node::build_mid(&primitives, 0);
        let root_node_sah = Node::build_sah(&primitives, 0);
        for root_node in &[root_node_mid, root_node_sah] {
            root_node.node_iterator_mut(
                &mut |s| for prim in &s.primitives {
                    assert!(prim.bounds.a.x >= s.bounds.a.x && prim.bounds.b.x <= s.bounds.b.x &&
                        prim.bounds.a.y >= s.bounds.a.y && prim.bounds.b.y <= s.bounds.b.y &&
                        prim.bounds.a.z >= s.bounds.a.z && prim.bounds.b.z <= s.bounds.b.z)
                }
            );
        }
    }

    #[test]
    fn test_subnodes_bound_within_node() {
        let mut scene = Scene::new();
        // load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        build_triangle_wall(Material::basic(), &mut scene, 0.5, 10.0);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };

        let root_node_mid = Node::build_mid(&primitives, 0);
        let root_node_sah = Node::build_sah(&primitives, 0);
        for root_node in &[root_node_mid, root_node_sah] {
            root_node.node_iterator(
                &|s| for prim in &s.primitives {
                    if !s.is_leaf {
                        let node_left_bounds = s.left.as_ref().unwrap().bounds;
                        let node_right_bounds = s.right.as_ref().unwrap().bounds;
                        assert!(node_left_bounds.a.x >= s.bounds.a.x && node_left_bounds.b.x <= s.bounds.b.x &&
                            node_left_bounds.a.y >= s.bounds.a.y && node_left_bounds.b.y <= s.bounds.b.y &&
                            node_left_bounds.a.z >= s.bounds.a.z && node_left_bounds.b.z <= s.bounds.b.z);
                        assert!(node_right_bounds.a.x >= s.bounds.a.x && node_right_bounds.b.x <= s.bounds.b.x &&
                            node_right_bounds.a.y >= s.bounds.a.y && node_right_bounds.b.y <= s.bounds.b.y &&
                            node_right_bounds.a.z >= s.bounds.a.z && node_right_bounds.b.z <= s.bounds.b.z);
                    }
                }
            );
        }
    }

    #[test]
    fn primitives_in_bvh_equal_scene() {
        let mut scene = Scene::new();
        // load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        build_triangle_wall(Material::basic(), &mut scene, 0.5, 10.0);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };
        let root_node_mid = Node::build_mid(&primitives, 0);
        let root_node_sah = Node::build_sah(&primitives, 0);
        for root_node in &[root_node_mid, root_node_sah] {
            let mut prim_count = 0;
            root_node.node_iterator_mut(&mut |s| if s.is_leaf { prim_count += s.primitives.len(); });
            assert_eq!(prim_count, primitives.len());
        }
    }

    #[test]
    fn primitive_midpoint_in_aabb() {
        let mut scene = Scene::new();
        // load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        build_triangle_wall(Material::basic(), &mut scene, 0.5, 10.0);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };
        let root_node_mid = Node::build_mid(&primitives, 0);
        let root_node_sah = Node::build_sah(&primitives, 0);
        for root_node in &[root_node_mid, root_node_sah] {
            let mut prim_count = 0;
            root_node.node_iterator_mut(&mut |s| if s.is_leaf { prim_count += s.primitives.len(); });
            assert_eq!(prim_count, primitives.len());
        }
    }

    #[test]
    fn always_hit_wall() {
        let mut scene = Scene::new();
        let offset = 10.0;
        let diff = 5.0;
        assert_eq!(scene.triangles.len(), 0);
        build_triangle_wall(Material::basic(), &mut scene, diff, offset);
        assert_eq!(scene.triangles.len(), ((2.0*offset/diff)*(2.0*offset/diff)*2.0) as usize);
        scene.generate_bvh_sah();
        println!("bounds: {:?}", scene.bvh.node.bounds);

        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };
        let root_node_sah = Node::build_sah(&primitives, 0);
        println!("Print tree:");
        scene.bvh.node.print(0);

        // generate a thousand rays with random
        let t = 1.0;
        let mut troublesome : Vec<Ray> = vec![];
        for i in 0..100000{
            println!("{:?}",i);
            let pos = Vec3 { x: 10.0 - 20.0*random::<f32>(), y: 10.0 - 20.0*random::<f32>(), z: 0.0 };
            let dir = Vec3 { x: random::<f32>(), y: random::<f32>(), z: random::<f32>() }.normalized_fast();
            let ray = Ray { pos: pos.subed(dir), dir };
            assert_small(ray.pos.added(ray.dir).z, 0.0);
            // let hit = inter_scene(ray, &scene);
            let mut hit = RayHit::NULL;
            let (aabb_hits, prim_hits, depth) = scene.bvh.intersect(ray, &scene, &mut hit);
            if ray.dir.z.abs() > EPSILON && (hit.t-t).abs() > EPSILON {
                troublesome.push(ray);
                println!("{:?}", i);
                // println!("{:?}", (aabb_hits, prim_hits, depth));
                println!("{:?}", hit.t);
                println!("{:?}", ray);
            }
        }
        for trouble in &troublesome {
            println!("{:?},", trouble);
        }
        assert_eq!(0, troublesome.len());
    }

    #[test]
    fn troublesome_hits_to_wall() {
        let mut scene = Scene::new();
        let offset = 10.0;
        let diff = 0.5;
        build_triangle_wall(Material::basic(), &mut scene, diff, offset);
        scene.generate_bvh_sah();
        println!("bounds: {:?}", scene.bvh.node.bounds);

        let rays = [
            // Ray { pos: Vec3 { x: 9.138742, y: 7.4507065, z: -0.0016354168 }, dir: Vec3 { x: 0.64129597, y: 0.7672918, z: 0.0016354168 } },
            // Ray { pos: Vec3 { x: 3.8617287, y: -5.4518433, z: -0.0018788258 }, dir: Vec3 { x: 0.8466592, y: 0.53213227, z: 0.0018788258 } },
            // Ray { pos: Vec3 { x: -6.3512807, y: -2.790472, z: -0.003994833 }, dir: Vec3 { x: 0.923883, y: 0.38265428, z: 0.003994833 } },
            Ray { pos: Vec3 { x: -5.4647856, y: 0.7814614, z: -0.00025426567 }, dir: Vec3 { x: 0.6266356, y: 0.77931243, z: 0.00025426567 } },
            // Ray { pos: Vec3 { x: -7.5424333, y: -6.336432, z: -0.0027151075 }, dir: Vec3 { x: 0.7013957, y: 0.71276695, z: 0.0027151075 } },
            // Ray { pos: Vec3 { x: -6.1725636, y: -2.1267533, z: -0.0015030894 }, dir: Vec3 { x: 0.9597282, y: 0.28092623, z: 0.0015030894 } },
            // Ray { pos: Vec3 { x: 5.731803, y: 7.9322295, z: -0.0008993151 }, dir: Vec3 { x: 0.7177717, y: 0.6962779, z: 0.0008993151 } },
            // Ray { pos: Vec3 { x: -7.5424333, y: -6.336432, z: -0.0027151075 }, dir: Vec3 { x: 0.7013957, y: 0.71276695, z: 0.0027151075 } },
        ];
        println!("inverted: {:?}", rays[0].inverted());

        for (i,ray) in rays.iter().enumerate() {
            // let hit = inter_scene(ray, &scene);
            let mut hit = RayHit::NULL;
            let (aabb_hits, prim_hits, depth) = scene.bvh.intersect(*ray, &scene, &mut hit);
            if ray.dir.z.abs() > EPSILON && (hit.t-1.0).abs() > EPSILON {
            // if hit.t != 0.0 && (hit.t-1.0).abs() > EPSILON {
                // println!("{:?}", i);
                // println!("{:?}", (aabb_hits, prim_hits, depth));
                println!("{:?}", ray);
                println!("{:?}", hit.t);
                // println!("{},{:?}", i, ray);
                // assert_small(hit.t, 1.0);
                panic!("expected not to get here");
            }
        }
    }

    fn neged_random() -> f32 {
        if random::<f32>() < 0.5 { -1.0 } else { 1.0 }
    }

    #[test]
    fn test_volume() {
        let mut scene = Scene::new();
        load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        scene.generate_bvh_sah();
        let mut custom_bounds = AABB::new();
        for i in 0..scene.triangles.len() {
            custom_bounds.a.x = custom_bounds.a.x.min(scene.triangles[i].a.x).min(scene.triangles[i].b.x).min(scene.triangles[i].c.x);
            custom_bounds.a.y = custom_bounds.a.y.min(scene.triangles[i].a.y).min(scene.triangles[i].b.y).min(scene.triangles[i].c.y);
            custom_bounds.a.z = custom_bounds.a.z.min(scene.triangles[i].a.z).min(scene.triangles[i].b.z).min(scene.triangles[i].c.z);
            custom_bounds.b.x = custom_bounds.b.x.max(scene.triangles[i].a.x).max(scene.triangles[i].b.x).max(scene.triangles[i].c.x);
            custom_bounds.b.y = custom_bounds.b.y.max(scene.triangles[i].a.y).max(scene.triangles[i].b.y).max(scene.triangles[i].c.y);
            custom_bounds.b.z = custom_bounds.b.z.max(scene.triangles[i].a.z).max(scene.triangles[i].b.z).max(scene.triangles[i].c.z);
        }

        assert_eq!(scene.bvh.node.bounds.a, custom_bounds.a.subed(Vec3::EPSILON));
        assert_eq!(scene.bvh.node.bounds.b, custom_bounds.b.added(Vec3::EPSILON));
    }

    #[test]
    fn test_intersection_logic() {
        let aabb = AABB{
            a: Vec3{ x: -10.0, y: -10.0, z: -10.0 },
            b: Vec3{ x:  10.0, y:  10.0, z:  10.0 }
        };
        // let t = 100*random::<f32>();
        // let mut troublesome : Vec<Ray> = vec![];
        // random
        for i in 0..100000 {
            //     println!("{:?}",i);
            let pos = Vec3 {
                x: neged_random() * (10.1 + 1000.0 * random::<f32>()),
                y: neged_random() * (10.1 + 1000.0 * random::<f32>()),
                z: 0.0
            };
            let dir = Vec3 { x: 100.0*random::<f32>(), y: 100.0*random::<f32>(), z: random::<f32>() }.normalized_fast();
        }
        //
        //
        //     let pos = Vec3 {
        //         x: neged_random()*(10.1+1000.0*random::<f32>()),
        //         y: neged_random()*(10.1+1000.0*random::<f32>()),
        //         z: neged_random()*(10.1+1000.0*random::<f32>())
        //     };
        //     let dir = Vec3 { x: random::<f32>(), y: random::<f32>(), z: random::<f32>() }.normalized_fast();
        //     let ray = Ray { pos: pos.subed(dir), dir };
    }
}
