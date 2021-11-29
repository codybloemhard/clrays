use crate::kernels::*;
use crate::scene::Scene;
use crate::info::Info;
use crate::cl_helpers::create_five;
use crate::misc::load_source;
use crate::cpu::{ whitted };
use crate::vec3::Vec3;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use ocl::{ Queue };
use std::f32::consts::{PI, FRAC_PI_2};

pub trait TraceProcessor{
    fn update(&mut self, events: &[Event]);
    fn render(&mut self) -> &[u32];
}

pub struct RealTracer{
    kernel: Box<TraceKernelReal>,
    queue: Queue,
}

impl RealTracer{
    pub fn new((width, height): (u32, u32), scene: &mut Scene, info: &mut Info) -> Result<Self, String>{
        let src = unpackdb!(load_source("assets/kernels/raytrace.cl"), "Could not load RealTracer's kernel!");
        info.set_time_point("Loading source file");
        let (_, _, _, program, queue) = unpackdb!(create_five(&src), "Could not init RealTracer's program and queue!");
        info.set_time_point("Creating OpenCL objects");
        let kernel = unpackdb!(TraceKernelReal::new("raytracing", (width, height), &program, &queue, scene, info), "Could not create RealTracer!");
        info.set_time_point("Last time stamp");
        Ok(RealTracer{
            kernel: Box::new(kernel),
            queue,
        })
    }
}

impl TraceProcessor for RealTracer{
    fn update(&mut self, _events: &[Event]){
        self.kernel.update(&self.queue).expect("Could not update RealTracer's kernel!");
    }

    fn render(&mut self) -> &[u32]{
        self.kernel.execute(&self.queue).expect("Could not execute RealTracer's kernel!");
        self.kernel.get_result(&self.queue).expect("Could not get result of RealTracer!")
    }
}

pub struct AaTracer{
    trace_kernel: Box<TraceKernelAa>,
    clear_kernel: Box<ClearKernel>,
    image_kernel: Box<ImageKernel>,
    queue: Queue,
}

impl AaTracer{
    pub fn new((width, height): (u32, u32), aa: u32, scene: &mut Scene, info: &mut Info) -> Result<Self, String>{
        let src = unpackdb!(load_source("assets/kernels/raytrace.cl"), "Could not load AaTracer's kernel!");
        info.set_time_point("Loading source file");
        let (_, _, _, program, queue) = unpackdb!(create_five(&src), "Could not init AaTracer's program and queue!");
        info.set_time_point("Creating OpenCL objects");
        let trace_kernel = unpackdb!(TraceKernelAa::new("raytracingAA", (width,height), aa, &program, &queue, scene, info), "Could not create AaTracer's trace kernel!");
        let clear_kernel = unpackdb!(ClearKernel::new("clear", (width,height), &program, &queue, trace_kernel.get_buffer_rc()), "Could not create AaTracer's clear kernel!");
        let image_kernel = unpackdb!(ImageKernel::new("image_from_floatmap", (width, height), &program, &queue, trace_kernel.get_buffer()), "Could not create AaTracer's image kernel!");
        info.set_time_point("Last time stamp");
        Ok(AaTracer{
            trace_kernel: Box::new(trace_kernel),
            clear_kernel: Box::new(clear_kernel),
            image_kernel: Box::new(image_kernel),
            queue,
        })
    }

}

impl TraceProcessor for AaTracer{
    fn update(&mut self, _events: &[Event]){
        self.trace_kernel.update(&self.queue).expect("Could not update AaTracer's trace kernel!");
    }

    fn render(&mut self) -> &[u32]{
        self.clear_kernel.execute(&self.queue).expect("Could not execute AaTracer's clear kernel!");
        self.trace_kernel.execute(&self.queue).expect("Could not execute AaTracer's trace kernel!");
        self.image_kernel.execute(&self.queue).expect("Could not execute AaTracer's image kernel!");
        self.image_kernel.get_result(&self.queue).expect("Could not get result of AaTracer's image kernel!")
    }
}

pub struct CpuWhitted<'a>{
    width: usize,
    height: usize,
    aa: usize,
    threads: usize,
    scene: &'a mut Scene,
    screen_buffer: Vec<u32>,
    float_buffer: Vec<Vec3>,
    texture_params: Vec<u32>,
    textures: Vec<u8>,
}

impl<'a> CpuWhitted<'a>{
    pub fn new(width: usize, height: usize, aa: usize, threads: usize, scene: &'a mut Scene, info: &mut Info) -> Self{
        let texture_params = scene.get_texture_params_buffer();
        info.set_time_point("Getting texture parameters");
        let textures = scene.get_textures_buffer();
        info.set_time_point("Getting texture buffer");
        let screen_buffer = vec![0; width * height];
        info.set_time_point("Creating screen buffer");
        let float_buffer = vec![Vec3::ZERO; width * height];
        Self{
            width,
            height,
            aa: aa.max(1),
            threads,
            scene,
            screen_buffer,
            float_buffer,
            texture_params,
            textures,
        }
    }
}

fn yaw_roll(yaw: f32, roll: f32) -> Vec3 {
    let a = roll;  // Up/Down
    let b = yaw;   // Left/Right
    Vec3 { x: a.cos() * b.sin(), y: a.sin(), z: -a.cos() * b.cos() }
}

impl TraceProcessor for CpuWhitted<'_>{
    fn update(&mut self, events: &[Event]){
        let cam = &mut self.scene.cam;
        for event in events.iter() {
            match event {
                Event::KeyDown { keycode: Some(Keycode::W), .. } => {
                    // Move Forward; Move into camera direction
                    let s = cam.move_sensitivity;
                    cam.pos.add(cam.dir.scaled(s));
                    break;
                },
                Event::KeyDown { keycode: Some(Keycode::S), .. } => {
                    // Move Backward; Move opposite camera direction
                    let s = cam.move_sensitivity;
                    cam.pos.add(cam.dir.neged().scaled(s));
                    break;
                },
                Event::KeyDown { keycode: Some(Keycode::D), .. } => {
                    // Move Right; Move camera direction crossed z-axis
                    let s = cam.move_sensitivity;
                    cam.pos.add(cam.dir.crossed(Vec3::UP).scaled(s));
                    break;
                },
                Event::KeyDown { keycode: Some(Keycode::A), .. } => {
                    // Move Left; Move camera direction crossed z-axis, negated
                    let s = cam.move_sensitivity;
                    cam.pos.add(cam.dir.crossed(Vec3::UP).neged().scaled(s));
                    break;
                },
                Event::KeyDown { keycode: Some(Keycode::I), .. } => {
                    // Look Up;
                    let s = cam.look_sensitivity;
                    cam.ori[1] = (cam.ori[1] + s).min(FRAC_PI_2).max(-FRAC_PI_2);
                    let yaw = cam.ori[0]; // Up/Down
                    let roll = cam.ori[1]; // Left/Right
                    cam.dir = yaw_roll(yaw, roll);
                },
                Event::KeyDown { keycode: Some(Keycode::K), .. } => {
                    // Look Down;
                    let s = cam.look_sensitivity;
                    cam.ori[1] = (cam.ori[1] - s).min(FRAC_PI_2).max(-FRAC_PI_2);
                    let yaw = cam.ori[0]; // Up/Down
                    let roll = cam.ori[1]; // Left/Right
                    cam.dir = yaw_roll(yaw, roll);
                },
                Event::KeyDown { keycode: Some(Keycode::L), .. } => {
                    // Look Right;
                    let s = cam.look_sensitivity;
                    cam.ori[0] += s;
                    if cam.ori[0] > PI {
                        cam.ori[0] -= 2.0 * PI;
                    }
                    let yaw = cam.ori[0]; // Up/Down
                    let roll = cam.ori[1]; // Left/Right
                    cam.dir = yaw_roll(yaw, roll);
                },
                Event::KeyDown { keycode: Some(Keycode::J), .. } => {
                    // Look Left;
                    let s = cam.look_sensitivity;
                    cam.ori[0] -= s;
                    if cam.ori[0] < -PI {
                        cam.ori[0] += 2.0 * PI;
                    }
                    let yaw = cam.ori[0]; // Up/Down
                    let roll = cam.ori[1]; // Left/Right
                    cam.dir = yaw_roll(yaw, roll);
                },
                _ => {}
            }
        }

    }

    fn render(&mut self) -> &[u32]{
        whitted(
            self.width, self.height, self.aa, self.threads,
            self.scene, &self.texture_params, &self.textures,
            &mut self.screen_buffer, &mut self.float_buffer,
        );
        &self.screen_buffer
    }
}
