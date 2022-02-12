use crate::scene::{ Scene, SceneItem, Material, Plane, Sphere };
use crate::vec3::{ Vec3 };
use crate::trace_tex::{ TexType };
use crate::consts::*;

pub fn gi_scene(scene: &mut Scene){
    scene.cam.pos = Vec3::new(0.0, 1.5, 6.0);
    scene.cam.dir = Vec3::BACKWARD;
    scene.cam.ori = Vec3::BACKWARD.normalized().orientation();

    scene.sky_col = Vec3::BLUE.unhardened(0.1);
    scene.add_texture("wood", "assets/textures/wood.png", TexType::Vector3c8bpc);
    scene.add_texture("sphere", "assets/textures/spheremap.jpg", TexType::Vector3c8bpc);
    scene.add_texture("stone-alb", "assets/textures/stone-albedo.png", TexType::Vector3c8bpc);
    scene.add_texture("stone-nor", "assets/textures/stone-normal.png", TexType::Vector3c8bpc);
    scene.add_texture("stone-rou", "assets/textures/stone-rough.png", TexType::Scalar8b);
    scene.add_texture("tiles-alb", "assets/textures/tiles-albedo.png", TexType::Vector3c8bpc);
    scene.add_texture("tiles-nor", "assets/textures/tiles-normal.png", TexType::Vector3c8bpc);
    scene.add_texture("tiles-rou", "assets/textures/tiles-rough.png", TexType::Scalar8b);
    scene.add_texture("scifi-alb", "assets/textures/scifi-albedo.png", TexType::Vector3c8bpc);
    scene.add_texture("scifi-nor", "assets/textures/scifi-normal.png", TexType::Vector3c8bpc);
    scene.add_texture("scifi-rou", "assets/textures/scifi-rough.png", TexType::Scalar8b);
    scene.add_texture("scifi-met", "assets/textures/scifi-metal.png", TexType::Scalar8b);
    scene.add_texture("solar-alb", "assets/textures/solar-albedo.png", TexType::Vector3c8bpc);
    scene.add_texture("solar-nor", "assets/textures/solar-normal.png", TexType::Vector3c8bpc);
    scene.add_texture("solar-rou", "assets/textures/solar-rough.png", TexType::Scalar8b);
    scene.add_texture("solar-met", "assets/textures/solar-metal.png", TexType::Scalar8b);
    scene.add_texture("sky", "assets/textures/sky0.jpg", TexType::Vector3c8bpc);
    scene.set_skybox("sky");
    scene.set_sky_intensity(10.0, 0.1, 2.0);

    Plane{
        pos: Vec3::new(0.0, -1.0, 0.0),
        nor: Vec3::UP,
        mat: Material::basic()
            //.as_conductor()
            //.with_specular(ALUMINIUM_SPEC)
            .as_dielectric()
            .with_roughness(0.1)
            .with_refraction(1.1)
            .with_texture(scene.get_texture("stone-alb"))
            .with_normal_map(scene.get_texture("stone-nor"))
            .with_roughness_map(scene.get_texture("stone-rou"))
            .with_tex_scale(4.0)
            .add_to_scene(scene)
    }.add(scene);

    Sphere{
        pos: Vec3::new(2.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_conductor()
            .with_roughness(0.5)
            .with_specular(COPPER_SPEC)
            .with_texture(scene.get_texture("tiles-alb"))
            .with_normal_map(scene.get_texture("tiles-nor"))
            .with_roughness_map(scene.get_texture("tiles-rou"))
            .add_to_scene(scene)
    }.add(scene);

    Sphere{
        pos: Vec3::new(0.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_conductor()
            .with_reflectivity(0.3)
            .with_roughness(0.1)
            .with_specular(GOLD_SPEC)
            .with_texture(scene.get_texture("solar-alb"))
            .with_normal_map(scene.get_texture("solar-nor"))
            .with_roughness_map(scene.get_texture("solar-rou"))
            .with_metalic_map(scene.get_texture("solar-met"))
            .add_to_scene(scene)
    }.add(scene);

    Sphere{
        pos: Vec3::new(-2.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_conductor()
            .with_roughness(0.02)
            .with_specular(Vec3{x: 0.001, y: 0.001, z: 0.002 })
            .with_texture(scene.get_texture("scifi-alb"))
            .with_normal_map(scene.get_texture("scifi-nor"))
            .with_roughness_map(scene.get_texture("scifi-rou"))
            .with_metalic_map(scene.get_texture("scifi-met"))
            .with_reflectivity(0.9)
            .add_to_scene(scene)
    }.add(scene);

    Sphere{
        pos: Vec3::new(-4.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_refraction(1.5)
            .with_roughness(0.01)
            .add_to_scene(scene)
    }.add(scene);

    Sphere{
        pos: Vec3::new(-6.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_refraction(2.0)
            .with_roughness(0.1)
            .with_colour(Vec3{ x: 0.8, y: 1.0, z: 0.7 })
            .add_to_scene(scene)
    }.add(scene);

    Sphere{
        pos: Vec3::new(0.0, 4.0, 3.0),
        rad: 2.0,
        mat: Material::basic().as_light(Vec3::uni(1.0), 10.0)
            .add_to_scene(scene)
    }.add(scene);
}
