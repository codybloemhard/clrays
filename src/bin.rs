#[macro_use]
extern crate clrays_rs;

use clrays_rs as clr;
use clr::window;
use clr::state;
use clr::scene::{ Scene, SceneItem, Material, Plane, Sphere, Light };
use clr::vec3::{ Vec3, BasicColour };
use clr::info::{ Info };
use clr::trace_tex::{ TexType };
use clr::trace_processor::{ TraceProcessor };

pub fn main() -> Result<(), String>{
    // clr::test(clr::test_platform::PlatformTest::OpenCl2);
    let mut info = Info::new();
    info.start_time();

    let mut scene = Scene::new();
    scene.sky_col = Vec3::soft_colour(BasicColour::Blue, 0.9, 0.2).normalized();
    scene.sky_intensity = 0.0;
    scene.cam_pos = Vec3::zero();
    scene.cam_dir = Vec3::backward();
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

    Plane{
        pos: Vec3::new(0.0, -1.0, 0.0),
        nor: Vec3::up(),
        mat: Material::basic()
            .with_texture(scene.get_texture("stone-alb"))
            .with_normal_map(scene.get_texture("stone-nor"))
            .with_roughness_map(scene.get_texture("stone-rou"))
            .with_tex_scale(4.0),
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(2.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_texture(scene.get_texture("tiles-alb"))
            .with_normal_map(scene.get_texture("tiles-nor"))
            .with_roughness_map(scene.get_texture("tiles-rou")),
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(0.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_reflectivity(0.3)
            .with_texture(scene.get_texture("solar-alb"))
            .with_normal_map(scene.get_texture("solar-nor"))
            .with_roughness_map(scene.get_texture("solar-rou"))
            .with_metalic_map(scene.get_texture("solar-met")),
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(-2.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic()
            .with_texture(scene.get_texture("scifi-alb"))
            .with_normal_map(scene.get_texture("scifi-nor"))
            .with_roughness_map(scene.get_texture("scifi-rou"))
            .with_metalic_map(scene.get_texture("scifi-met"))
            .with_reflectivity(0.5),
    }.add(&mut scene);

    Light{
        pos: Vec3::new(0.0, 2.0, -3.0),
        intensity: 100.0,
        col: Vec3::one(),
    }.add(&mut scene);

    info.set_time_point("Setting up scene");
    scene.pack_textures(&mut info);

    //let (w,h) = (960u32,540u32);
    //let (w,h) = (1600u32,900u32);
    let (w,h) = (1920u32,1080u32);
    //let tracer = unpackdb!(TraceProcessor::new_real((w, h), &mut scene, &mut info), "Could not create TraceProcessor");
    let tracer = unpackdb!(TraceProcessor::new_aa((w, h), 2, &mut scene, &mut info), "Could not create TraceProssor");

    info.stop_time();
    info.print_info();

    let mut window = window::Window::<state::StdState>::new("ClRays", w, h, tracer);
    window.run(window::std_input_handler);
    Ok(())
}