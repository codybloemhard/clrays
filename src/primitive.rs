use crate::scene::{Sphere, Plane, Scene, Triangle, Model, Intersectable};
use crate::vec3::Vec3;
use crate::consts::EPSILON;
use crate::cpu::inter::{ Ray, RayHit};

#[derive(Copy, Clone, Debug)]
pub enum Shape {
    MODEL,
    SPHERE,
    TRIANGLE,
    PLANE
}

#[derive(Clone, Copy)]
pub struct Primitive {
    pub shape_type: Shape,
    pub index: usize,
}
impl Primitive {

    pub fn from_model(model: Model) -> Self {
        Self {
            // bounds: aabb,
            shape_type: Shape::MODEL,
            index: model.mesh as usize
        }
    }

    pub fn from_sphere(sphere: &Sphere, index_sphere: usize) -> Self{
        Self {
            shape_type: Shape::SPHERE,
            index: index_sphere,
        }
    }

    pub fn from_plane(plane: Plane, index_plane: usize) -> Self{
        // todo: check normal
        assert!(plane.nor.dot(Vec3::UP).abs().le(&EPSILON) ||
                plane.nor.dot(Vec3::RIGHT).abs().le(&EPSILON) ||
                plane.nor.dot(Vec3::FORWARD).abs().le(&EPSILON));
        Self {
            shape_type: Shape::PLANE,
            index: index_plane,
        }
    }

    pub fn from_triangle(triangle: Triangle, index_triangle: usize) -> Self{
        Self {
            shape_type: Shape::TRIANGLE,
            index: index_triangle,
        }
    }

    pub fn intersect(&self, ray: Ray, scene: &Scene, hit: &mut RayHit) -> (usize,usize) {
        match self.shape_type {
            Shape::MODEL => {
                let model = scene.models[self.index];
                let t = hit.t;
                // transform ray
                let ori = model.rot.orientation();
                let yaw = ori.yaw;
                let mut ray = ray;
                ray.pos = ray.pos.subed(model.pos);
                // rotate pos clockwise x-axis
                ray.pos = ray.pos.yawed(-yaw);
                // rotate dir clockwise x-axis
                ray.dir = ray.dir.yawed(-yaw);
                let (a,b) = scene.sub_bvhs[model.mesh as usize].intersect_mesh(ray, scene, hit);
                if hit.t < t { // apply
                    hit.pos.added(model.pos);
                    hit.mat = model.mat;
                    hit.nor = hit.nor.yawed(yaw);
                }
                return (a,b);
            },
            Shape::SPHERE => scene.spheres[self.index].intersect(ray, hit),
            Shape::PLANE => scene.planes[self.index].intersect(ray, hit),
            Shape::TRIANGLE => scene.triangles[self.index].intersect(ray, hit),
        }
        (0,1)
    }
}
