#[macro_use]
extern crate clrays_rs;

use clrays_rs as clr;
use clr::window;
use clr::trace_processor;
use clr::scene::{ Scene, Camera, SceneItem, Material, Plane, Triangle, Sphere, Light };
use clr::vec3::{ Vec3 };
use clr::info::{ Info };
use clr::trace_tex::{ TexType };
use clr::state::{ State, Settings, log_update_fn, fps_input_fn };
use clr::mesh::load_model;

use sdl2::keyboard::Keycode;
use clrays_rs::consts::*;

pub const USE_WATERFLOOR : bool = true;
pub const USE_WIDE_ANGLE : bool = false;

pub fn main() -> Result<(), String>{
    // clr::test(clr::test_platform::PlatformTest::OpenCl2);
    let mut info = Info::new();
    info.start_time();

    let mut scene = Scene::new();
    scene.cam = Camera{
        // pos: Vec3::new(0.0, 5.0, -8.0),
        // dir: Vec3::new(0.0, -1.0, 2.0).normalized(),
        pos: Vec3::ZERO,
        dir: Vec3::BACKWARD,
        ori: [0.0, 0.0],
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

    if USE_WATERFLOOR {
        Plane{
            pos: Vec3::new(0.0, -4.0, 0.0),
            nor: Vec3::UP,
            mat: Material::basic()
                .as_checkerboard()
                .with_texture(scene.get_texture("stone-alb"))
                .with_normal_map(scene.get_texture("stone-nor"))
                .with_roughness_map(scene.get_texture("stone-rou"))
                .with_tex_scale(4.0),
        }.add(&mut scene);

        Plane{
            pos: Vec3::new(0.0, -1.0, 0.0),
            nor: Vec3::UP,
            mat: Material::basic()
                .as_dielectric()
                .with_absorption(WATER_ABSORPTION)
                .with_refraction(WATER_REFRACTION)
        }.add(&mut scene);
    } else {
        Plane{
            pos: Vec3::new(0.0, -1.0, 0.0),
            nor: Vec3::UP,
            mat: Material::basic()
                .as_checkerboard()
                .with_texture(scene.get_texture("stone-alb"))
                .with_normal_map(scene.get_texture("stone-nor"))
                .with_roughness_map(scene.get_texture("stone-rou"))
                .with_tex_scale(4.0),
        }.add(&mut scene);
    }

    Sphere{
        pos: Vec3::new(2.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .with_texture(scene.get_texture("tiles-alb"))
            .with_normal_map(scene.get_texture("tiles-nor"))
            .with_roughness_map(scene.get_texture("tiles-rou")),
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(0.0, 0.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .with_reflectivity(0.3)
            .with_texture(scene.get_texture("solar-alb"))
            .with_normal_map(scene.get_texture("solar-nor"))
            .with_roughness_map(scene.get_texture("solar-rou"))
            .with_metalic_map(scene.get_texture("solar-met")),
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(-2.0, 0.1, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .with_texture(scene.get_texture("scifi-alb"))
            .with_normal_map(scene.get_texture("scifi-nor"))
            .with_roughness_map(scene.get_texture("scifi-rou"))
            .with_metalic_map(scene.get_texture("scifi-met"))
            .with_reflectivity(0.9),
    }.add(&mut scene);

    Triangle{
        a: Vec3::new(-1.0, 1.0, -7.0),
        b: Vec3::new( 1.0, 1.0, -7.0),
        c: Vec3::new( 1.0, 3.0, -7.0),
        mat: Material::basic().as_checkerboard(),
    }.add(&mut scene);

    // https://groups.csail.mit.edu/graphics/classes/6.837/F03/models/
    // load_model("assets/models/teapot.obj", Material::basic(), &mut scene);

    Sphere{
        pos: Vec3::new(3.0, 3.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_refraction(0.7)
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(-0.0, 3.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_refraction(1.6)
    }.add(&mut scene);
    Sphere{
        pos: Vec3::new(-0.0, 3.0, -5.0),
        rad: 0.95 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(-3.0, 3.0, -5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_refraction(2.0)
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(0.0, 2.0, 5.0),
        rad: 1.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_absorption(Vec3 { x: 0.8, y: 0.3, z: 0.3 })
            .with_refraction(AIR_REFRACTION)
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(-3.0, 2.0, 5.0),
        rad: 2.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_absorption(Vec3 { x: 0.8, y: 0.3, z: 0.3 })
            .with_refraction(AIR_REFRACTION)
    }.add(&mut scene);

    Sphere{
        pos: Vec3::new(-10.0, 5.0, 5.0),
        rad: 5.0 - EPSILON,
        mat: Material::basic()
            .as_dielectric()
            .with_absorption(Vec3 { x: 0.8, y: 0.3, z: 0.3 })
            .with_refraction(AIR_REFRACTION)
    }.add(&mut scene);

    Light{
        pos: Vec3::new(0.0, 2.0, -3.0),
        intensity: 100.0,
        col: Vec3::ONE,
    }.add(&mut scene);

    info.set_time_point("Setting up scene");
    scene.pack_textures(&mut info);

    let settings = Settings{
        aa_samples: 12,
        max_reduced_ms: 40.0,
        start_in_focus_mode: false,
    };
    // let mut state = State::new(build_keymap!(W, S, A, D, Q, E, I, K, J, L, U, O), settings);
    let mut state = State::new(build_keymap!(M, T, S, N, G, L, U, E, A, O, F, B), settings);

    // let (w, h) = (960, 540);
    // let (w, h) = (1600, 900);
    let (w, h) = (1920, 1080);

    // let mut tracer = unpackdb!(trace_processor::RealTracer::new((w, h), &mut scene, &mut info), "Could not create RealTracer!");
    // let mut tracer = unpackdb!(trace_processor::AaTracer::new((w, h), 2, &mut scene, &mut info), "Could not create AaTracer!");
    let mut tracer = trace_processor::CpuWhitted::new(w as usize, h as usize, 32, &mut scene, &mut info);

    info.stop_time();
    info.print_info();

    let mut window = window::Window::new("ClRays", w as u32, h as u32);
    window.run(fps_input_fn, log_update_fn, &mut state, &mut tracer, &mut scene)
}
