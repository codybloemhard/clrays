use crate::aabb::{AABB, Axis};
use crate::scene::{Scene, Model, Intersectable};
use crate::cpu::inter::{Ray, RayHit, dist_sphere, dist_triangle};
use crate::vec3::Vec3;

#[derive(Copy, Clone, Debug)]
pub enum Shape {
    MODEL = 0,
    SPHERE = 1,
    TRIANGLE = 2,
    PLANE = 3,
}

#[derive(Clone, Copy)]
pub struct Primitive {
    pub shape_type: Shape,
    pub index: usize,
}

impl Intersectable for Primitive {
    fn vertices(&self) -> Vec<Vec3> {
        unimplemented!()
    }

    fn intersect(&self, ray: Ray, hit: &mut RayHit) {
        unimplemented!()
    }

    #[inline]
    fn clip(&self, aabb: AABB, self_bound: &AABB) -> AABB{
        aabb.overlap(*self_bound)
    }
}

impl Primitive {
    pub fn from_model(model: Model) -> Self {
        Self {
            // bounds: aabb,
            shape_type: Shape::MODEL,
            index: model.mesh as usize
        }
    }

    pub fn from_sphere(index_sphere: usize) -> Self{
        Self {
            shape_type: Shape::SPHERE,
            index: index_sphere,
        }
    }

    pub fn from_triangle(index_triangle: usize) -> Self{
        Self {
            shape_type: Shape::TRIANGLE,
            index: index_triangle,
        }
    }

    pub fn intersect(&self, ray: Ray, scene: &Scene, hit: &mut RayHit) -> (usize, usize){
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
                let (a, b) = scene.sub_bvhs[model.mesh as usize].intersect(ray, scene, hit);
                if hit.t < t { // apply
                    hit.mat = model.mat;
                    hit.nor = hit.nor.yawed(yaw);

                    hit.pos = hit.pos.yawed(yaw);
                    hit.pos = hit.pos.added(model.pos);
                }
                return (a, b);
            },
            Shape::SPHERE => scene.spheres[self.index].intersect(ray, hit),
            Shape::TRIANGLE => scene.triangles[self.index].intersect(ray, hit),
            Shape::PLANE => unimplemented!()
        }
        (0, 1)
    }

    pub fn occluded(&self, ray: Ray, scene: &Scene, dist: f32) -> bool{
        match self.shape_type {
            Shape::MODEL => {
                let model = scene.models[self.index];
                // transform ray
                let ori = model.rot.orientation();
                let yaw = ori.yaw;
                let mut ray = ray;
                ray.pos = ray.pos.subed(model.pos);
                // rotate pos clockwise x-axis
                ray.pos = ray.pos.yawed(-yaw);
                // rotate dir clockwise x-axis
                ray.dir = ray.dir.yawed(-yaw);
                scene.sub_bvhs[model.mesh as usize].occluded(ray, scene, dist)
            },
            Shape::SPHERE => dist_sphere(ray, &scene.spheres[self.index]) <= dist,
            Shape::TRIANGLE => dist_triangle(ray, &scene.triangles[self.index]) <= dist,
            Shape::PLANE => unimplemented!()
        }
    }
}
