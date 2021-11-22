use crate::kernels::*;
use crate::scene::{ Scene };
use crate::info::{ Info };
use crate::cl_helpers::{ create_five };
use crate::misc::{ load_source };

use ocl::{ Queue };

pub trait TraceProcessor{
    fn update(&mut self);
    fn render(&mut self) -> &[i32];
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
    fn update(&mut self){
        self.kernel.update(&self.queue).expect("Could not update RealTracer's kernel!");
    }

    fn render(&mut self) -> &[i32]{
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
    fn update(&mut self){
        self.trace_kernel.update(&self.queue).expect("Could not update AaTracer's trace kernel!");
    }

    fn render(&mut self) -> &[i32]{
        self.clear_kernel.execute(&self.queue).expect("Could not execute AaTracer's clear kernel!");
        self.trace_kernel.execute(&self.queue).expect("Could not execute AaTracer's trace kernel!");
        self.image_kernel.execute(&self.queue).expect("Could not execute AaTracer's image kernel!");
        self.image_kernel.get_result(&self.queue).expect("Could not get result of AaTracer's image kernel!")
    }
}
