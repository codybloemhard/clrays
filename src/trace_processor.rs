use crate::kernels::*;
use crate::scene::Scene;
use crate::info::Info;
use crate::cl_helpers::create_five;
use crate::misc::load_source;
use crate::cpu::{ whitted };
use crate::vec3::Vec3;
use crate::state::{ RenderMode, State };
use crate::config::ConfigParsed;

use ocl::{ Queue };

use rand::prelude::*;

pub trait TraceProcessor{
    fn update(&mut self, scene: &mut Scene, state: &State);
    fn render(&mut self, scene: &mut Scene, state: &mut State) -> &[u32];
}

pub struct GpuWhitted{
    kernel: Box<TraceKernelWhitted>,
    queue: Queue,
}

impl GpuWhitted{
    pub fn new((width, height): (u32, u32), scene: &mut Scene, info: &mut Info) -> Result<Self, String>{
        let src = unpackdb!(load_source("assets/kernels/raytrace.cl"), "Could not load GpuWhitted's kernel!");
        info.set_time_point("Loading source file");
        let (_, _, _, program, queue) = unpackdb!(create_five(&src), "Could not init GpuWhitted's program and queue!");
        info.set_time_point("Creating OpenCL objects");
        let kernel = unpackdb!(TraceKernelWhitted::new("raytracing", (width, height), &program, &queue, scene, info), "Could not create GpuWhitted!");
        info.set_time_point("Last time stamp");
        Ok(Self{
            kernel: Box::new(kernel),
            queue,
        })
    }
}

impl TraceProcessor for GpuWhitted{
    fn update(&mut self, scene: &mut Scene, _: &State){
        self.kernel.update(&self.queue, scene).expect("Could not update GpuWhitted's kernel!");
    }

    fn render(&mut self, _: &mut Scene, state: &mut State) -> &[u32]{
        match state.render_mode{
            RenderMode::Full | RenderMode::Reduced => {
                state.last_frame = RenderMode::Full;
                state.render_mode = RenderMode::None;
                self.kernel.execute(&self.queue).expect("Could not execute GpuWhitted's kernel!");
                self.kernel.get_result(&self.queue).expect("Could not get result of GpuWhitted!")
            },
            _ => {
                state.last_frame = RenderMode::None;
                self.kernel.get_result(&self.queue).expect("Could not get result of GpuWhitted!")
            },
        }
    }
}

pub struct GpuPath{
    trace_kernel: Box<TraceKernelPath>,
    image_kernel: Box<ImageKernel>,
    clear_kernel: Box<ClearKernel>,
    queue: Queue,
}

impl GpuPath{
    pub fn new((width, height): (u32, u32), scene: &mut Scene, conf: &ConfigParsed, info: &mut Info) -> Result<Self, String>{
        let src = unpackdb!(load_source("assets/kernels/raytrace.cl"), "Could not load GpuPath's kernel!");
        info.set_time_point("Loading source file");
        let (_, _, _, program, queue) = unpackdb!(create_five(&src), "Could not init GpuPath's program and queue!");
        info.set_time_point("Creating OpenCL objects");
        let trace_kernel = unpackdb!(TraceKernelPath::new("pathtracing", (width, height), &program, &queue, scene, info), "Could not create GpuPath's trace kernel!");
        let tm = conf.post.tone_map as u32;
        let image_kernel = unpackdb!(ImageKernel::new("image_from_floatmap", (width, height), tm, &program, &queue, trace_kernel.get_buffer()), "Could not create GpuPath's image kernel!");
        let clear_kernel = unpackdb!(ClearKernel::new("clear", (width, height), &program, &queue, trace_kernel.get_buffer()), "Could not create GpuPath's clear kernel!");
        info.set_time_point("Last time stamp");
        Ok(Self{
            trace_kernel: Box::new(trace_kernel),
            image_kernel: Box::new(image_kernel),
            clear_kernel: Box::new(clear_kernel),
            queue,
        })
    }

}

impl TraceProcessor for GpuPath{
    fn update(&mut self, scene: &mut Scene, state: &State){
        self.trace_kernel.update(&self.queue, scene, state).expect("Could not update GpuPath's trace kernel!");
    }

    fn render(&mut self, _: &mut Scene, state: &mut State) -> &[u32]{
        state.last_frame = RenderMode::Full;
        state.render_mode = RenderMode::Full;
        if state.moved{
            self.clear_kernel.execute(&self.queue).expect("Could not execute GpuPath's clearn kernel!");
            state.samples_taken = 0;
        }
        self.trace_kernel.execute(&self.queue).expect("Could not execute GpuPath's trace kernel!");
        state.samples_taken += 1;
        if state.settings.calc_frame_energy{
            state.frame_energy = self.trace_kernel.frame_energy(&self.queue) / state.samples_taken as f32;
        }
        self.image_kernel.set_divider(state.samples_taken as f32).expect("Could not set GpuPath's clear kernel's divider argument!");
        self.image_kernel.execute(&self.queue).expect("Could not execute GpuPath's image kernel!");
        self.image_kernel.get_result(&self.queue).expect("Could not get result of GpuPath's image kernel!")
    }
}

pub struct CpuWhitted{
    width: usize,
    height: usize,
    threads: usize,
    screen_buffer: Vec<u32>,
    float_buffer: Vec<Vec3>,
    texture_params: Vec<u32>,
    textures: Vec<u8>,
    rng: ThreadRng,
}

impl CpuWhitted{
    pub fn new(width: usize, height: usize, threads: usize, scene: &mut Scene, info: &mut Info) -> Self{
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
            threads,
            screen_buffer,
            float_buffer,
            texture_params,
            textures,
            rng: rand::thread_rng(),
        }
    }
}

impl TraceProcessor for CpuWhitted{
    fn update(&mut self, _: &mut Scene, _: &State){ }

    fn render(&mut self, scene: &mut Scene, state: &mut State) -> &[u32]{
        whitted(
            self.width, self.height, self.threads,
            scene, &self.texture_params, &self.textures,
            &mut self.screen_buffer, &mut self.float_buffer, state, &mut self.rng
        );
        &self.screen_buffer
    }
}
