use crate::vec3::Vec3;
use crate::consts::{EPSILON, UV_SPHERE, UV_PLANE};
use crate::scene::{Sphere, Plane, Triangle, Scene};
use crate::cpu::{Ray, RayHit};

// ray-sphere intersection
#[inline]
fn inter_sphere<'a>(ray: Ray, sphere: &'a Sphere, closest: &mut RayHit<'a>){
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
fn inter_plane<'a>(ray: Ray, plane: &'a Plane, closest: &mut RayHit<'a>){
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
fn inter_triangle<'a>(ray: Ray, tri: &'a Triangle, closest: &mut RayHit<'a>){
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
        // Vec3 {
        //     x: 0.5 * (self.a.x + self.b.x),
        //     y: 0.5 * (self.a.y + self.b.y),
        //     z: 0.5 * (self.a.z + self.b.z),
        // }
        self.a.added(self.b).scaled(0.5)
    }

    // [source](http://www.pbr-book.org/3ed-2018/Shapes/Basic_Shape_Interface.html#Bounds3::IntersectP)
    pub fn intersection(&self, ray: Ray) -> Option<(f32,f32)> {
        let inv_dir = Vec3 {
            x: if ray.dir.x.abs() > EPSILON { 1.0 / ray.dir.x } else { f32::MAX },
            y: if ray.dir.y.abs() > EPSILON { 1.0 / ray.dir.y } else { f32::MAX },
            z: if ray.dir.z.abs() > EPSILON { 1.0 / ray.dir.z } else { f32::MAX },
        };
        let dir_is_neg : [bool; 3] = [
            ray.dir.x < 0.0,
            ray.dir.y < 0.0,
            ray.dir.z < 0.0,
        ];

        let mut t_min;
        let mut t_max;
        let ss = [&self.a, &self.b];

        // Compute intersections with x and y slabs.
        let tx_min = (ss[  dir_is_neg[0] as usize].x - ray.pos.x) * inv_dir.x;
        let tx_max = (ss[1-dir_is_neg[0] as usize].x - ray.pos.x) * inv_dir.x;
        let ty_min = (ss[  dir_is_neg[1] as usize].y - ray.pos.y) * inv_dir.y;
        let ty_max = (ss[1-dir_is_neg[1] as usize].y - ray.pos.y) * inv_dir.y;

        // Check intersection within x and y bounds.
        if (tx_min > ty_max) || (tx_max < ty_min) {
            return None;
        }
        t_min = if ty_min > tx_min { ty_min } else { tx_min };
        t_max = if ty_max < tx_max { ty_max } else { tx_max };

        // Compute intersections z slab.
        let tz_min = (ss[  dir_is_neg[2] as usize].z - ray.pos.z) * inv_dir.z;
        let tz_max = (ss[1-dir_is_neg[2] as usize].z - ray.pos.z) * inv_dir.z;

        // Check intersection within x and y and z bounds.
        if (t_min > tz_max) || (t_max < tz_min) {
            return None;
        }
        t_min = if tz_min > t_min { tz_min } else { t_min };
        t_max = if tz_max < t_max { tz_max } else { t_max };

        if tz_min > t_min { t_min = tz_min; }
        if tz_max < t_max { t_max = tz_max; }

        Some((t_min, t_max))
    }

    pub fn hits(&self, ray: Ray) -> bool {
        if let Some((t_min, t_max)) = self.intersection(ray) {
            true
        } else {
            false
        }
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
    fn new() -> Self {
        Self {
            bounds: AABB::new(),
            shape_type: Shape::NONE,
            sphere: 0,
            plane: 0,
            triangle: 0,
        }
    }

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
        Self {
            bounds: AABB::new(), // HELP: worth supporting plane? If the normal does not align with an axis its AABB is infinite...?
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
    pub fn intersect<'a>(&self, ray: Ray, scene: &'a Scene, closest: &mut RayHit<'a>) {
        if self.bounds.hits(ray) {
            if self.is_leaf {
                for primitive in self.primitives.iter() {
                    primitive.intersect(ray, scene, closest);
                }
            } else {
                self.left.as_ref().unwrap().intersect(ray, scene, closest);
                self.right.as_ref().unwrap().intersect(ray, scene, closest);
            }
        }
    }

    pub fn print(&self, depth: usize) {
        println!("test");
        println!("{}", format!("{:>width$}", self.get_primitives_count(), width = 2*depth));
        if !self.is_leaf {
            println!("{}", format!("{:>width$}", self.get_primitives_count(), width = 2*depth));
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
    for i in 0..scene.planes.len() {
        primitives.push(Primitive::from_plane(&scene.planes[i], i));
    };
    for i in 0..scene.triangles.len() {
        primitives.push(Primitive::from_triangle(&scene.triangles[i], i));
    };

    let root_node = build_subnode(&primitives, 0);
    BVH {
        node: root_node
    }
}

fn build_subnode(primitives: &Vec<Primitive>, depth: usize) -> Node {

    println!("dep: {}", depth);
    println!("len: {}", primitives.len());

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
            println!("{}, {}", first, tmp);
        }
        first = tmp;
        val += tmp;
    }
    val /= primitives.len() as f32;

    // Decide whether we apply the primitives into a leaf node
    if primitives.len() < 5 || is_all_the_same {
        println!("Returnin early");
        let node = Node {
            bounds,
            primitives: primitives.clone(),
            is_leaf: true,
            left: None,
            right: None
        };
        return node;
    }

    println!("mid_val: {}", val);
    println!("bounds: {:?}", bounds);

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
        // println!("bounds: {:?}; midpoint: {:?}; val: {}", primitive.bounds, primitive.bounds.midpoint(), val);
        // println!("bounds: {:?}", primitive.bounds);
        println!("midpoint: {:?}", primitive.bounds.midpoint());
    }
    println!("left: {}", left.len());
    println!("right: {}", right.len());

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
    pub fn intersect<'a>(&self, ray: Ray, scene: &'a Scene, closest: &mut RayHit<'a>) {
        self.node.intersect(ray, scene, closest);
    }
}