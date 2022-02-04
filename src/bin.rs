#[macro_use]
extern crate clrays_rs;

use clrays_rs as clr;
use clr::window;
use clr::trace_processor;
use clr::scene::{ Scene, SceneType, Camera, SceneItem, Material, Plane, Sphere, Light, Model, /*Triangle*/ };
use clr::vec3::{ Vec3 };
use clr::info::{ Info };
use clr::trace_tex::{ TexType };
use clr::state::{ State, Settings, log_update_fn, fps_input_fn };
use clr::consts::*;
use clr::vec3::Orientation;

use sdl2::keyboard::Keycode;
use std::env;
use std::process::exit;

pub const USE_WIDE_ANGLE: bool = false;

pub fn main() -> Result<(), String>{
    // clr::test(clr::test_platform::PlatformTest::OpenCl2);
    let mut info = Info::new();
    info.start_time();

    let mut scene = Scene::new();

    fn select_target() {
        println!("Run program with:");
        println!("cargo run --release -- cpu");
        println!("or");
        println!("cargo run --release -- gpu");
        exit(0)
    }

    let args: Vec<String> = env::args().collect();
    if args.len() == 2 {
        let target = &args[1];
        if target == "cpu" { scene.stype = SceneType::Whitted }
        else if *target == "gpu" { scene.stype = SceneType::GI }
        else { select_target() }
    } else { select_target() }

    scene.cam = Camera{
        pos: Vec3::new(0.0, 1.5, 6.0),
        dir: Vec3::BACKWARD,
        ori: Vec3::BACKWARD.normalized().orientation(),
        move_sensitivity: 0.1,
        look_sensitivity: 0.05,
        fov: 80.0,
        chromatic_aberration_shift: 2,
        chromatic_aberration_strength: 0.3,
        vignette_strength: 0.1,
        angle_radius: if USE_WIDE_ANGLE { FRAC_2_PI } else { 0.0 },
        distortion_coefficient: 2.0
    };

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
            .as_conductor()
            .with_roughness(0.2)
            .with_specular(ALUMINIUM_SPEC)
            .with_texture(scene.get_texture("stone-alb"))
            .with_normal_map(scene.get_texture("stone-nor"))
            .with_roughness_map(scene.get_texture("stone-rou"))
            .with_tex_scale(4.0)
            .add_to_scene(&mut scene)
    }.add(&mut scene);

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
            .add_to_scene(&mut scene)
    }.add(&mut scene);

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
            .add_to_scene(&mut scene)
    }.add(&mut scene);

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
            .add_to_scene(&mut scene)
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(-4.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_refraction(1.5)
            .with_roughness(0.01)
            .add_to_scene(&mut scene)
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(-6.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_refraction(2.0)
            .with_roughness(0.1)
            .with_colour(Vec3{ x: 0.8, y: 1.0, z: 0.7 })
            .add_to_scene(&mut scene)
    }.add(&mut scene);

    // Sphere{
    //     pos: Vec3::new(-6.0, 0.0, -5.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .with_refraction(1.6)
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    // Sphere{
    //     pos: Vec3::new(-6.0, 0.0, -5.0),
    //     rad: 0.95 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);

    // Sphere{
    //     pos: Vec3::new(0.0, 2.0, -10.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .with_absorption(Vec3 { x: 0.8, y: 0.3, z: 0.3 })
    //         .with_refraction(DIAMOND_REFRACTION)
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    //
    // Sphere{
    //     pos: Vec3::new(-3.0, 2.0, -10.0),
    //     rad: 2.0 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .with_absorption(Vec3 { x: 0.8, y: 0.3, z: 0.3 })
    //         .with_refraction(AIR_REFRACTION)
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    //
    // Sphere{
    //     pos: Vec3::new(-10.0, 5.0, -10.0),
    //     rad: 5.0 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .with_absorption(Vec3 { x: 0.8, y: 0.3, z: 0.3 })
    //         .with_refraction(AIR_REFRACTION)
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);

    let mut dragon = Model{
        pos: Default::default(),
        rot: Default::default(),
        mat: Material::basic().with_colour(Vec3::new(1.0, 0.5, 0.4)).add_to_scene(&mut scene),
        // mat: Material::basic().as_dielectric().with_refraction(WATER_REFRACTION).add_to_scene(&mut scene),
        mesh: scene.add_mesh("assets/models/dragon.obj".parse().unwrap())
    };
    // 10000 dragons = 1 billion triangles
    for _ in 0..0 {
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
        intensity: 10000.0,
        col: Vec3::ONE,
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(0.0, 4.0, 3.0),
        rad: 2.0,
        mat: Material::basic().as_light(Vec3::uni(1.0), 10.0)
            .add_to_scene(&mut scene)
    }.add(&mut scene);

    scene.gen_top_bvh();

    info.set_time_point("Setting up scene");
    scene.pack_textures(&mut info);

    let settings = Settings{
        aa_samples: 8,
        max_reduced_ms: 40.0,
        start_in_focus_mode: false,
        max_render_depth: 4,
        calc_frame_energy: false,
    };
    // let mut state = State::new(build_keymap!(W, S, A, D, Q, E, I, K, J, L, U, O, T), settings);
    let mut state = State::new(build_keymap!(M, T, S, N, G, L, U, E, A, O, F, B, W), settings);

    // let (w, h) = (960, 540);
    // let (w, h) = (1600, 900);
    let (w, h) = (1920, 1080);

    let mut window = window::Window::new("ClRays", w as u32, h as u32);
    match scene.stype {
        SceneType::GI => {
            let mut tracer_gpu = unpackdb!(trace_processor::GpuPath::new((w, h), &mut scene, &mut info), "Could not create GpuPath!");
            // let mut tracer_gpu = unpackdb!(trace_processor::GpuWhitted::new((w, h), &mut scene, &mut info), "Could not create GpuPath!");
            info.stop_time();
            info.print_info();
            window.run(fps_input_fn, log_update_fn, &mut state, &mut tracer_gpu, &mut scene)
        },
        SceneType::Whitted => {
            let mut tracer_cpu = trace_processor::CpuWhitted::new(w as usize, h as usize, 32, &mut scene, &mut info);
            info.stop_time();
            info.print_info();
            window.run(fps_input_fn, log_update_fn, &mut state, &mut tracer_cpu, &mut scene)
        }
    }
}
