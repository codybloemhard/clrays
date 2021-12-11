use crate::vec3::Vec3;
use crate::consts::{EPSILON, UV_SPHERE, UV_PLANE};
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
    if a > -EPSILON && a < EPSILON { return; } // ray parallel to tri
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
        self.a.added(self.b).scaled(0.5)
    }

    // [source](http://www.pbr-book.org/3ed-2018/Shapes/Basic_Shape_Interface.html#Bounds3::IntersectP)
    // pub fn intersection(&self, ray: Ray, inv_dir: Vec3, dir_is_neg: [bool; 3]) -> Option<(f32,f32)> {
    pub fn intersection(&self, ray: Ray, inv_dir: Vec3, dir_is_neg: [usize; 3]) -> Option<(f32,f32)> {
        let ss = [&self.a, &self.b];

        // Compute intersections with x and y slabs.
        let mut t_min  = (ss[  dir_is_neg[0]].x - ray.pos.x) * inv_dir.x;
        let mut t_max  = (ss[1-dir_is_neg[0]].x - ray.pos.x) * inv_dir.x;
        let mut ty_min = (ss[  dir_is_neg[1]].y - ray.pos.y) * inv_dir.y;
        let mut ty_max = (ss[1-dir_is_neg[1]].y - ray.pos.y) * inv_dir.y;

        // t_min *= 1.0 + 2.0 * gamma(3);
        // t_max *= 1.0 + 2.0 * gamma(3);
        // ty_min *= 1.0 + 2.0 * gamma(3);
        // ty_max *= 1.0 + 2.0 * gamma(3);

        // Check intersection within x and y bounds.
        if (t_min > ty_max) || (t_max < ty_min) {
            // println!("(t_min > ty_max) || (t_max < ty_min)");
            return None;
        }
        t_min = t_min.max(ty_min);
        t_max = t_max.min(ty_max);

        // Compute intersections z slab.
        let mut tz_min = (ss[  dir_is_neg[2]].z - ray.pos.z) * inv_dir.z;
        let mut tz_max = (ss[1-dir_is_neg[2]].z - ray.pos.z) * inv_dir.z;
        // println!("{},{}", ray.pos.z, inv_dir.z);
        // println!("{},{}", tz_min, tz_max);

        // tz_min *= 1.0 + 2.0 * gamma(3);
        // tz_max *= 1.0 + 2.0 * gamma(3);

        // Check intersection within x and y and z bounds.
        if (t_min > tz_max) || (t_max < tz_min) {
            // println!("(t_min > tz_max) || (t_max < tz_min)");
            return None;
        }
        t_min = t_min.max(tz_min);
        t_max = t_max.min(tz_max);

        if t_min > t_max || t_min == f32::INFINITY{
            // println!("t_min > t_max");
            return None;
        }
        Some((t_min, t_max))
    }

    pub fn volume(self) -> f32{
        let v = self.b.subed(self.a);
        v.x * v.y * v.z
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
enum Axis {
    X,
    Y,
    Z
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
    // fn new() -> Self {
    //     Self {
    //     bounds: AABB::new(),
    //         shape_type: Shape::NONE,
    //         sphere: 0,
    //         plane: 0,
    //         triangle: 0,
    //     }
    // }

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
                    x: -5000.0, // Using f32::INFINITY crashes my computer
                    y: plane.pos.y,
                    z: -5000.0 // Using f32::INFINITY crashes my computer
                },
                b: Vec3 {
                    x: 5000.0, // Using f32::INFINITY crashes my computer
                    y: plane.pos.y + EPSILON,
                    z: 5000.0, // Using f32::INFINITY crashes my computer
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
                },
                b : Vec3 {
                    x: triangle.a.x.max(triangle.b.x).max(triangle.c.x),
                    y: triangle.a.y.max(triangle.b.y).max(triangle.c.y),
                    z: triangle.a.z.max(triangle.b.z).max(triangle.c.z),
                }
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

pub struct Node {
    pub bounds: AABB,  // AABB: 24 bytes
    pub is_leaf: bool, // bool: 1 bit
    pub primitives: Vec<Primitive>,
    pub left: Option<Box<Node>>,
    pub right: Option<Box<Node>>,
}

impl Node {
    pub fn node_iterator(self, call_on_every_node: &dyn Fn(&Node)) {
        call_on_every_node(&self);
        if !self.is_leaf {
            let node_left = self.left.unwrap();
            let node_right = self.right.unwrap();

            node_left.node_iterator(call_on_every_node);
            node_right.node_iterator(call_on_every_node);
        }
    }

    pub fn node_iterator_mut(self, call_on_every_node: &mut dyn FnMut(&Node)) {
        call_on_every_node(&self);
        if !self.is_leaf {
            let node_left = self.left.unwrap();
            let node_right = self.right.unwrap();

            node_left.node_iterator_mut(call_on_every_node);
            node_right.node_iterator_mut(call_on_every_node);
        }
    }

    pub fn intersect<'a>(&self, ray: Ray, scene: &'a Scene, closest: &mut RayHit<'a>, inv_dir: Vec3, dir_is_neg: [usize; 3] ) -> (usize, usize, usize) {
        if self.is_leaf {
            for primitive in self.primitives.iter() {
                primitive.intersect(ray, scene, closest);
            }
            (0, self.primitives.len(), 1)
        } else {
            // Check which box hits first
            let node_left = self.left.as_ref().unwrap();
            let node_right = self.right.as_ref().unwrap();

            let intersection_left = node_left.bounds.intersection(ray, inv_dir, dir_is_neg);
            let intersection_right = node_right.bounds.intersection(ray, inv_dir, dir_is_neg);

            let (tl0, tl1) = intersection_left.unwrap_or((f32::INFINITY, f32::INFINITY));
            let (tr0, tr1) = intersection_right.unwrap_or((f32::INFINITY, f32::INFINITY));

            let tl = if let Some((tl0, tl1)) = intersection_left{
                if tl0 > 0.0 && tl0 < f32::MAX { tl0 }
                else if tl1 > 0.0 && tl1 < f32::MAX { tl1 }
                else { f32::MAX }
            } else { f32::MAX };
            let tr = if let Some((t1r0, tr1)) = intersection_right{
                if tr0 > 0.0 && tr0 < f32::MAX { tr0 }
                else if tr1 > 0.0 && tr1 < f32::MAX { tr1 }
                else { f32::MAX }
            } else { f32::MAX };

            let mut x1 = (0, 0, 0);
            let mut x2 = (0, 0, 0);

            if tl <= tr && tl < f32::MAX {
                // First intersect left
                x1 = node_left.intersect(ray, scene, closest, inv_dir, dir_is_neg);
                if tr < closest.t && tr < f32::MAX {
                    x2 = node_right.intersect(ray, scene, closest, inv_dir, dir_is_neg);
                }
            }
            else if tr < tl && tr < f32::MAX{
                // First intersect right
                x2 = node_right.intersect(ray, scene, closest, inv_dir, dir_is_neg);
                if tl < closest.t && tl < f32::MAX {
                    x1 = node_left.intersect(ray, scene, closest, inv_dir, dir_is_neg);
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

pub fn build_bvh(scene: &Scene) -> BVH {
    let mut primitives: Vec<Primitive> = vec![];

    // Build primitives
    for i in 0..scene.spheres.len() {
        primitives.push(Primitive::from_sphere(&scene.spheres[i], i));
    };
    // for i in 0..scene.planes.len() {
    //     primitives.push(Primitive::from_plane(&scene.planes[i], i));
    // };
    for i in 0..scene.triangles.len() {
        primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
    };

    let root_node = build_subnode(&primitives, 0);
    BVH {
        node: root_node
    }
}

fn build_subnode(primitives: &Vec<Primitive>, depth: usize) -> Node {

    // Find bounds
    let mut bounds = AABB::new();
    for primitive in primitives.iter() {
        bounds.a.x = bounds.a.x.min( primitive.bounds.a.x );
        bounds.a.y = bounds.a.y.min( primitive.bounds.a.y );
        bounds.a.z = bounds.a.z.min( primitive.bounds.a.z );

        bounds.b.x = bounds.b.x.max( primitive.bounds.b.x );
        bounds.b.y = bounds.b.y.max( primitive.bounds.b.y );
        bounds.b.z = bounds.b.z.max( primitive.bounds.b.z );
    }

    // Find dominant axis
    let diff_x = bounds.b.x - bounds.a.x;
    let diff_y = bounds.b.y - bounds.a.y;
    let diff_z = bounds.b.z - bounds.a.z;
    let axis = if diff_x > diff_y && diff_x > diff_z {
        Axis::X
    } else if diff_y > diff_z {
        Axis::Y
    } else {
        Axis::Z
    };

    // Find midpoint
    let mut val: f32 = 0.0;
    let mut first : f32 = f32::MIN;
    let mut is_all_the_same = true;
    for primitive in primitives.iter() {
        let tmp = match axis {
            Axis::X => primitive.bounds.midpoint().x,
            Axis::Y => primitive.bounds.midpoint().y,
            Axis::Z => primitive.bounds.midpoint().z,
        };
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
    for primitive in primitives.iter() {
        if match axis {
            Axis::X => primitive.bounds.midpoint().x,
            Axis::Y => primitive.bounds.midpoint().y,
            Axis::Z => primitive.bounds.midpoint().z,
        } < val {
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
        left: Some(Box::new(build_subnode(&left, depth + 1))),
        right: Some(Box::new(build_subnode(&right, depth + 1)))
    };
    return node;
}

impl BVH {
    pub fn intersect<'a>(&self, ray: Ray, scene: &'a Scene, closest: &mut RayHit<'a>) -> (usize, usize, usize) {
        let inv_dir = ray.inverted().dir;
        let dir_is_neg : [usize; 3] = ray.direction_negations();
        self.node.intersect(ray, scene, closest, inv_dir, dir_is_neg)
    }
}


#[cfg(test)]
mod test {
    use crate::cpu::{Ray, RayHit};
    use crate::vec3::Vec3;
    use crate::bvh::{AABB, Primitive, build_subnode};
    use crate::consts::EPSILON;
    use crate::scene::{Triangle, Material, Scene};
    use crate::mesh::load_model;

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
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert!(intersection.is_some());
        let (t0, t1) = intersection.unwrap();
        assert!(t0 < 0.0 && t0 > -2.0 && t1 > 0.0 && t1 < 2.0);
        assert!(t0 < t1);

        // Before
        let ray = Ray {
            pos: Vec3::ONE.neged().scaled(5.0),
            dir: Vec3::ONE.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert!(intersection.is_some());
        let (t0, t1) = intersection.unwrap();
        println!("{},{}", t0,t1);
        assert!(t0 > 5.0 && t1 > 9.0 && t1 < 11.0);
        assert!(t0 < t1);

        // After
        let ray = Ray {
            pos: Vec3::ONE.scaled(5.0),
            dir: Vec3::ONE.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        assert!(intersection.is_some());
        let (t0, t1) = intersection.unwrap();
        assert!(t1 < -5.0 && t0 < -9.0 && t0 > -11.0);
        assert!(t0 < t1);

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
        let (t0, t1) = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations()).unwrap();
        assert_small(t0, -1.0);
        assert_small(t1,  1.0);

        // Before
        let ray = Ray {
            pos: Vec3::FORWARD.neged().scaled(5.0),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let (t0, t1) = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations()).unwrap();
        assert_small(t0, 4.0);
        assert_small(t1, 6.0);

        // After
        let ray = Ray {
            pos: Vec3::FORWARD.scaled(5.0),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let (t0, t1) = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations()).unwrap();
        assert_small(t0, -6.0);
        assert_small(t1, -4.0);

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
        assert!(intersection.is_none());

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
        assert!(intersection.is_none());

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
        assert!(intersection.is_none());
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
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        let (t0, t1) = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations()).unwrap();
        assert_small(t0, -1.0);
        assert_small(t1,  1.0);

        // Before
        let ray = Ray {
            pos: Vec3::FORWARD.neged().scaled(5.0).added(Vec3::UP ),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        let (t0, t1) = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations()).unwrap();
        assert_small(t0, 4.0);
        assert_small(t1, 6.0);

        // After
        let ray = Ray {
            pos: Vec3::FORWARD.scaled(5.0).added(Vec3::UP ),
            dir: Vec3::FORWARD.normalized()
        };
        let aabb = AABB {
            a: Vec3::ONE.neged(),
            b: Vec3::ONE
        };
        let intersection = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations());
        let (t0, t1) = aabb.intersection(ray, ray.inverted().dir, ray.direction_negations()).unwrap();
        assert_small(t0, -6.0);
        assert_small(t1, -4.0);
    }

    #[test]
    fn bounding_box_triangle() {
        let triangle = Triangle{
            a: Vec3{ x: 0.0, y: 0.0, z: 0.0 },
            b: Vec3{ x: 1.0, y: 0.0, z: 0.0 },
            c: Vec3{ x: 0.0, y: 1.0, z: 0.0 },
            mat: Material::basic()
        };
        let prim = Primitive::from_triangle(&triangle, 0);
        assert!(prim.bounds.a.equal(&triangle.a));
        assert!(prim.bounds.b.equal(&triangle.b.added(triangle.c)));
    }

    #[test]
    fn ensure_aabb_has_volume(){
        let mut scene = Scene::new();
        load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };
        let root_node = build_subnode(&primitives, 0);
        root_node.node_iterator(&|s| assert!(s.bounds.volume() > 0.0) );
    }

    #[test]
    fn test_all_primitives_bound_within_node() {
        let mut scene = Scene::new();
        load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };
        let root_node = build_subnode(&primitives, 0);
        root_node.node_iterator(
            &|s| for prim in &s.primitives {
                assert!(prim.bounds.a.x >= s.bounds.a.x && prim.bounds.b.x <= s.bounds.b.x &&
                    prim.bounds.a.y >= s.bounds.a.y && prim.bounds.b.y <= s.bounds.b.y &&
                    prim.bounds.a.z >= s.bounds.a.z && prim.bounds.b.z <= s.bounds.b.z)
            }
        );
    }

    #[test]
    fn test_subnodes_bound_within_node() {
        let mut scene = Scene::new();
        load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };
        let root_node = build_subnode(&primitives, 0);
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

    #[test]
    fn primitives_in_bvh_equal_scene() {
        let mut scene = Scene::new();
        load_model("assets/models/teapot.obj", Material::basic(), &mut scene);
        let mut primitives: Vec<Primitive> = vec![];
        for i in 0..scene.triangles.len() {
            primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
        };
        let root_node = build_subnode(&primitives, 0);
        let mut prim_count = 0;
        root_node.node_iterator_mut( &mut |s| if s.is_leaf { prim_count += s.primitives.len(); } );
        assert_eq!(prim_count, primitives.len());
    }
}
