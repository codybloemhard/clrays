#[macro_use]
extern crate clrays_rs;

use clrays_rs as clr;
use clr::window;
use clr::trace_processor;
use clr::scene::{ Scene, RenderType };
use clr::info::{ Info };
use clr::state::{ State, Settings, log_update_fn, fps_input_fn };
use clr::scenes::{ gi_scene::gi_scene, whitted_scene::whitted_scene };

use sdl2::keyboard::Keycode;
use std::env;
use std::process::exit;

pub fn main() -> Result<(), String>{
    // clr::test(clr::test_platform::PlatformTest::OpenCl2);
    let mut info = Info::new();
    info.start_time();

    let mut scene = Scene::new();

    fn select_target_msg() {
        println!("Run program with:");
        println!("cargo run --release -- cpu");
        println!("or");
        println!("cargo run --release -- gpu");
        exit(0)
    }

    let args: Vec<String> = env::args().collect();
    if args.len() == 2 {
        let target = &args[1];
        if target == "cpu" { scene.stype = RenderType::Whitted }
        else if *target == "gpu" { scene.stype = RenderType::GI }
        else { select_target_msg() }
    } else { select_target_msg() }

    let render_type = scene.stype;

    match render_type{
        RenderType::GI => gi_scene(&mut scene),
        RenderType::Whitted => whitted_scene(&mut scene),
    }

    scene.gen_top_bvh();

    info.set_time_point("Setting up scene");
    scene.pack_textures(&mut info);

    let settings = Settings{
        aa_samples: 8,
        max_reduced_ms: 40.0,
        start_in_focus_mode: false,
        max_render_depth: 4,
        calc_frame_energy: false,
        render_type: render_type,
    };

    // let mut state = State::new(build_keymap!(W, S, A, D, Q, E, I, K, J, L, U, O, T), settings);
    let mut state = State::new(build_keymap!(M, T, S, N, G, L, U, E, A, O, F, B, W), settings);

    // let (w, h) = (960, 540);
    // let (w, h) = (1600, 900);
    let (w, h) = (1920, 1080);

    let mut window = window::Window::new("ClRays", w as u32, h as u32);
    match render_type {
        RenderType::GI => {
            let mut tracer_gpu = unpackdb!(trace_processor::GpuPath::new((w, h), &mut scene, &mut info), "Could not create GpuPath!");
            // let mut tracer_gpu = unpackdb!(trace_processor::GpuWhitted::new((w, h), &mut scene, &mut info), "Could not create GpuPath!");
            info.stop_time();
            info.print_info();
            window.run(fps_input_fn, log_update_fn, &mut state, &mut tracer_gpu, &mut scene)
        },
        RenderType::Whitted => {
            let mut tracer_cpu = trace_processor::CpuWhitted::new(w as usize, h as usize, 32, &mut scene, &mut info);
            info.stop_time();
            info.print_info();
            window.run(fps_input_fn, log_update_fn, &mut state, &mut tracer_cpu, &mut scene)
        }
    }
}
