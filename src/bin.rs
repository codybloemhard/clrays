#[macro_use]
extern crate clrays_rs;

use clrays_rs as clr;
use clr::window;
use clr::trace_processor;
use clr::scene::{ Scene, RenderType };
use clr::info::{ Info };
use clr::state::{ State, Settings, log_update_fn, fps_input_fn };
use clr::scenes::{ gi_scene::gi_scene, whitted_scene::whitted_scene };
use clr::config::Config;

use sdl2::keyboard::Keycode;

use std::env;
use std::path::Path;

pub fn main() -> Result<(), String>{
    let mut info = Info::new();
    info.start_time();

    let args: Vec<String> = env::args().collect();
    let conf = if args.len() == 2 {
        let conf = &args[1];
        Config::read(Path::new(conf)).expect("Could not read config!")
    } else {
        panic!("Please pass a toml configuration file as an argument!");
    };
    let conf = conf.parse().expect("Could not parse config!");

    if let Some(title) = conf.base.title.clone(){
        println!("Loaded config with title: '{}'", title);
    } else {
        println!("Loaded config!");
    }

    let render_type = if let Some(rt) = conf.base.render_type{ rt }
    else {
        clr::test(clr::test_platform::PlatformTest::OpenCl0);
        clr::test(clr::test_platform::PlatformTest::OpenCl1);
        clr::test(clr::test_platform::PlatformTest::OpenCl2);
        clr::test(clr::test_platform::PlatformTest::SdlAudio);
        clr::test(clr::test_platform::PlatformTest::SdlWindow);
        return Ok(());
    };


    let mut scene = Scene::new(&conf);
    scene.stype = render_type;

    match render_type{
        RenderType::GI => gi_scene(&mut scene),
        RenderType::Whitted => whitted_scene(&mut scene),
    }

    scene.gen_top_bvh();

    info.set_time_point("Setting up scene");
    scene.pack_textures(&mut info);

    let settings = Settings{
        aa_samples: conf.cpu.aa_samples,
        max_reduced_ms: conf.cpu.max_reduced_ms,
        start_in_focus_mode: conf.cpu.start_in_focus_mode,
        max_render_depth: conf.cpu.render_depth,
        calc_frame_energy: conf.base.frame_energy,
        render_type: render_type,
    };

    // let mut state = State::new(build_keymap!(W, S, A, D, Q, E, I, K, J, L, U, O, T), settings);
    let mut state = State::new(build_keymap!(M, T, S, N, G, L, U, E, A, O, F, B, W), settings);

    let mut window = window::Window::new("ClRays", conf.base.w, conf.base.h);

    macro_rules! run{
        ($tracer:ident) => {
            info.stop_time();
            info.print_info();
            return window.run(fps_input_fn, log_update_fn, &mut state, &mut $tracer, &mut scene);
        }
    }

    match (conf.base.gpu, render_type){
        (true, RenderType::GI) => {
            let mut tracer_gpu = unpackdb!(trace_processor::GpuPath::new((conf.base.w, conf.base.h), &mut scene, &mut info), "Could not create GpuPath!");
            run!(tracer_gpu);
        },
        (true, RenderType::Whitted) => {
            let mut tracer_gpu = unpackdb!(trace_processor::GpuWhitted::new((conf.base.w, conf.base.h), &mut scene, &mut info), "Could not create GpuPath!");
            run!(tracer_gpu);
        },
        (false, RenderType::Whitted) => {
            let mut tracer_cpu = trace_processor::CpuWhitted::new(conf.base.w as usize, conf.base.h as usize, 32, &mut scene, &mut info);
            run!(tracer_cpu);
        },
        (on_gpu, rtype) => {
            panic!("Combination of {} and {:?} is not supported!", if on_gpu { "gpu" } else { "cpu" }, rtype);
        },
    }
}
