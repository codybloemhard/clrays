#[macro_use]
extern crate clrays_rs;

use clrays_rs as clr;
use clr::window;
use clr::trace_processor;
use clr::scene::{ Scene, Camera, Material, Plane, Triangle, Sphere, Light };
use clr::vec3::{ Vec3 };
use clr::info::{ Info };
use clr::trace_tex::{ TexType };
use clr::state::{ State, Settings, log_update_fn, fps_input_fn };

use sdl2::keyboard::Keycode;
use clrays_rs::consts::*;
use clrays_rs::scene::{Model, SceneItem};
use clrays_rs::vec3::Orientation;
use clrays_rs::bvh::Bvh;
use clrays_rs::mesh::Mesh;
use stopwatch::Stopwatch;
// use clrays_rs::mesh::build_triangle_wall;

pub const USE_WATERFLOOR : bool = false;
pub const USE_WIDE_ANGLE : bool = false;

// credit: George Marsaglia
#[inline]
fn xor32(seed: &mut u32) -> u32{
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

pub fn main() -> Result<(), String>{
    // clr::test(clr::test_platform::PlatformTest::OpenCl2);
    let mut info = Info::new();
    info.start_time();

    let mut scene = Scene::new();
    scene.cam = Camera{
        // pos: Vec3::new(0.0, 5.0, -8.0),
        // dir: Vec3::new(0.0, -1.0, 2.0).normalized(),
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

    if USE_WATERFLOOR {
        Plane{
            pos: Vec3::new(0.0, -4.0, 0.0),
            nor: Vec3::UP,
            mat: Material::basic()
                .as_checkerboard()
                .with_texture(scene.get_texture("stone-alb"))
                .with_normal_map(scene.get_texture("stone-nor"))
                .with_roughness_map(scene.get_texture("stone-rou"))
                .with_tex_scale(4.0)
                .add_to_scene(&mut scene)
        }.add(&mut scene);

        Plane{
            pos: Vec3::new(0.0, -1.0, 0.0),
            nor: Vec3::UP,
            mat: Material::basic()
                .as_dielectric()
                .with_absorption(WATER_ABSORPTION)
                .with_refraction(WATER_REFRACTION)
                .add_to_scene(&mut scene)
        }.add(&mut scene);
    } else {
        Plane{
            pos: Vec3::new(0.0, -1.0, 0.0),
            nor: Vec3::UP,
            mat: Material::basic()
                // .as_checkerboard()
                .with_texture(scene.get_texture("stone-alb"))
                .with_normal_map(scene.get_texture("stone-nor"))
                .with_roughness_map(scene.get_texture("stone-rou"))
                .with_tex_scale(4.0)
                .add_to_scene(&mut scene)
        }.add(&mut scene);
    }
    //
    // Sphere{
    //     pos: Vec3::new(2.0, 0.0, -5.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .with_texture(scene.get_texture("tiles-alb"))
    //         .with_normal_map(scene.get_texture("tiles-nor"))
    //         .with_roughness_map(scene.get_texture("tiles-rou"))
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    //
    // Sphere{
    //     pos: Vec3::new(0.0, 0.0, -5.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .with_reflectivity(0.3)
    //         .with_texture(scene.get_texture("solar-alb"))
    //         .with_normal_map(scene.get_texture("solar-nor"))
    //         .with_roughness_map(scene.get_texture("solar-rou"))
    //         .with_metalic_map(scene.get_texture("solar-met"))
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    //
    // Sphere{
    //     pos: Vec3::new(-2.0, 0.1, -5.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .with_texture(scene.get_texture("scifi-alb"))
    //         .with_normal_map(scene.get_texture("scifi-nor"))
    //         .with_roughness_map(scene.get_texture("scifi-rou"))
    //         .with_metalic_map(scene.get_texture("scifi-met"))
    //         .with_reflectivity(0.9)
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);

    // Triangle{
    //     a: Vec3::new(-1.0, 1.0, -7.0),
    //     b: Vec3::new( 1.0, 1.0, -7.0),
    //     c: Vec3::new( 1.0, 3.0, -7.0),
    //     mat: Material::basic().as_checkerboard().add_to_scene(&mut scene),
    // }.add(&mut scene);

    // Sphere{
    //     pos: Vec3::new(3.0, 3.0, -5.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .with_refraction(0.7)
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    //
    // Sphere{
    //     pos: Vec3::new(-0.0, 3.0, -5.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .with_refraction(1.6)
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    // Sphere{
    //     pos: Vec3::new(-0.0, 3.0, -5.0),
    //     rad: 0.95 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    //
    // Sphere{
    //     pos: Vec3::new(-3.0, 3.0, -5.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .with_refraction(2.0)
    //         .add_to_scene(&mut scene)
    // }.add(&mut scene);
    //
    // Sphere{
    //     pos: Vec3::new(0.0, 2.0, -10.0),
    //     rad: 1.0 - EPSILON,
    //     mat: Material::basic()
    //         .as_dielectric()
    //         .with_absorption(Vec3 { x: 0.8, y: 0.3, z: 0.3 })
    //         .with_refraction(AIR_REFRACTION)
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
    // // println!("build wall...");
    // // build_triangle_wall(Material::basic(), &mut scene, 0.5, 10.0,);
    // // println!("wall is build");

    // https://groups.csail.mit.edu/graphics/classes/6.837/F03/models/
    // load_model("assets/models/object-scene.obj", Material::basic(), &mut scene);
    // let model_mat = Material::basic();
            // .with_reflectivity(0.3);
            // .as_dielectric()
            // .with_refraction(2.0);
    // load_model("assets/models/teapot.obj", model_mat.add_to_scene(&mut scene), &mut scene);

    // generate 5.000.000 triangles randomly
    let mut triangles = vec![];
    let mut seed:u32 = 81349324; // guaranteed to be random

    for i in 0..5000000{
        if i % 100000 == 0 {
            println!("{}",i);
        }
        triangles.push(Triangle{
            a: Vec3 {
                x: xor32(&mut seed) as f32,
                y: xor32(&mut seed) as f32,
                z: xor32(&mut seed) as f32
            },
            b: Vec3 {
                x: xor32(&mut seed) as f32,
                y: xor32(&mut seed) as f32,
                z: xor32(&mut seed) as f32
            },
            c: Vec3 {
                x: xor32(&mut seed) as f32,
                y: xor32(&mut seed) as f32,
                z: xor32(&mut seed) as f32
            },
        });
        // println!("{},{:?}",i, triangles[i]);
    }
    println!("building bvh...");
    // generate bvh over triangles
    let watch = Stopwatch::start_new();
    Bvh::from_mesh(Mesh::default(), &triangles, 12);
    let mut elapsed = watch.elapsed_ms();
    println!("done building bvh in {}...", elapsed);
    return Err("".parse().unwrap());

    let mut dragon = Model{
        pos: Default::default(),
        rot: Default::default(),
        mat: scene.add_material(Material::basic()),
        mesh: scene.add_mesh("assets/models/dragon.obj".parse().unwrap())
    };
    let n = 20;
    for i in 0..n {
        println!("{}",i);
        let theta = (i as f32 / n as f32) * 2.0*PI;
        println!("{}",theta);
        println!("{}",FRAC_2_PI);
        let mut pos = Vec3::BACKWARD.scaled(16.0).yawed(theta).added(Vec3::FORWARD.scaled(6.0)).subed(Vec3::UP.scaled(5.0));
        // pos = pos.subed(Vec3::UP.scaled(3.0));
        println!("{:?}",pos);
        let mut ori = Vec3::ZERO.subed(pos).orientation();
        println!("{}",theta);
        println!("{}",FRAC_2_PI);
        ori.yaw = theta + FRAC_2_PI ;
        // ori.yaw = FRAC_2_PI;
        // assert_eq!(ori.yaw, FRAC_2_PI );
        println!("{:?}",ori);
        dragon.pos = pos;
        dragon.rot = Vec3::from_orientation( &ori);
        scene.add_model(dragon);
    }

    Light{
        pos: Vec3::new(0.0, 3.0, -5.0),
        intensity: 100.0,
        col: Vec3::ONE,
    }.add(&mut scene);

    // println!("build bvh...");
    // scene.generate_bvh_sah();
    // scene.generate_bvh_mid();
    // println!("bvh is build");
    // scene.bvh.node.print(0);
    // let mut v0 = Vec::new();
    // scene.bvh.node.get_prim_counts(&mut v0);
    // println!("{:?}, {}, {}", v0, v0.len(), v0.iter().sum::<usize>());

    // scene.generate_bvh_nightly(16);
    // let mut v1 = Vec::new();
    // scene.bvh_nightly.get_prim_counts(0, &mut v1);
    // println!("{:?}, {}, {}", v1, v1.len(), v1.iter().sum::<usize>());

    // println!("{:?}, {:?}", scene.bvh.node.bounds, scene.bvh_nightly.vertices[0].bound);
    // println!("{:?}, {:?}", scene.bvh.node.left.as_ref().unwrap().bounds, scene.bvh_nightly.vertices[2].bound);
    // println!("{:?}, {:?}", scene.bvh.node.right.as_ref().unwrap().bounds, scene.bvh_nightly.vertices[3].bound);

    info.set_time_point("Setting up scene");
    scene.pack_textures(&mut info);

    let settings = Settings{
        aa_samples: 8,
        max_reduced_ms: 40.0,
        start_in_focus_mode: false,
        max_render_depth: 4,
    };
    // let mut state = State::new(build_keymap!(W, S, A, D, Q, E, I, K, J, L, U, O, T), settings);
    let mut state = State::new(build_keymap!(M, T, S, N, G, L, U, E, A, O, F, B, W), settings);

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
