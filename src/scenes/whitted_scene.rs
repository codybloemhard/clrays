use crate::scene::{ Scene, SceneItem, Light, Model, };
use crate::material::Material;
use crate::vec3::{ Vec3 };
use crate::consts::*;
use crate::vec3::Orientation;

pub const USE_WIDE_ANGLE: bool = false;

pub fn whitted_scene(scene: &mut Scene){
    scene.cam.pos = Vec3::new(0.0, 1.5, 6.0);
    scene.cam.dir = Vec3::BACKWARD;
    scene.cam.ori = Vec3::BACKWARD.normalized().orientation();

    let mut dragon = Model{
        pos: Default::default(),
        rot: Default::default(),
        mat: Material::basic().with_colour(Vec3::new(1.0, 0.5, 0.4)).add_to_scene(scene),
        // mat: Material::basic().as_dielectric().with_refraction(WATER_REFRACTION).add_to_scene(scene),
        mesh: scene.add_mesh("assets/models/dragon.obj".parse().unwrap())
    };

    // 10000 dragons = 1 billion triangles
    for _ in 0..5 {
        let rad = 20.0;
        let pos = Vec3 {
            x: rand::random::<f32>() * rad - rad * 0.5,
            y: rand::random::<f32>() * rad - rad * 0.5,
            z: rand::random::<f32>() * rad - rad * 0.5,
        };
        let mut ori = Orientation { yaw: 0.0, roll: 0.0 };
        let theta = rand::random::<f32>() * 2.0 * PI;
        ori.yaw = theta;
        dragon.pos = pos;
        dragon.rot = Vec3::from_orientation( &ori);
        scene.add_model(dragon);
    }

    Light{
        pos: Vec3::new(0.0, 3.0, 0.0),
        intensity: 1000.0,
        col: Vec3::ONE,
    }.add(scene);
}
