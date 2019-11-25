use ocl::{Queue};
use crate::kernels::*;
use crate::scene::{Scene};
use crate::info::{Info};
use crate::cl_helpers::{create_five};
use crate::misc::{load_source};

pub enum TraceProcessor{
    RealTracer(TraceKernelReal,Queue),
    AaTracer(TraceKernelAa,ClearKernel,ImageKernel,Queue)
}

impl TraceProcessor{
    pub fn new_real((width,height): (u32,u32), scene: &mut Scene, info: &mut Info) -> Result<Self,String>{
        info.start_time();
        let src = unpackdb!(load_source("../Assets/Kernels/raytrace.cl"));
        info.set_time_point("Loading source file");
        let (_,_,_,program,queue) = unpackdb!(create_five(&src));
        info.set_time_point("Creating OpenCL objects");
        let kernel = unpackdb!(TraceKernelReal::new("raytracing", (width,height), &program, &queue, scene, info));
        info.set_time_point("Last time stamp");
        info.stop_time();
        info.print_info();
        Ok(TraceProcessor::RealTracer(kernel,queue))
    }

    pub fn new_aa((width,height): (u32,u32), aa: u32, scene: &mut Scene, info: &mut Info) -> Result<Self,String>{
        info.start_time();
        let src = unpackdb!(load_source("../Assets/Kernels/raytrace.cl"));
        info.set_time_point("Loading source file");
        let (_,_,_,program,queue) = unpackdb!(create_five(&src));
        info.set_time_point("Creating OpenCL objects");
        let kernel = unpackdb!(TraceKernelAa::new("raytracingAA", (width,height), aa, &program, &queue, scene, info));
        let clear_kernel = unpackdb!(ClearKernel::new("clear", (width,height), &program, &queue, kernel.get_buffer_rc()));
        let img_kernel = unpackdb!(ImageKernel::new("image_from_floatmap", (width,height), &program, &queue, kernel.get_buffer()));
        info.set_time_point("Last time stamp");
        info.stop_time();
        info.print_info();
        Ok(TraceProcessor::AaTracer(kernel,clear_kernel,img_kernel,queue))
    }

    pub fn update(&mut self) -> Result<(),ocl::Error>{
        match self{
            TraceProcessor::RealTracer(kernel,queue) => kernel.update(queue),
            TraceProcessor::AaTracer(kernel,_,_,queue) => kernel.update(queue),
        }
    }

    pub fn render(&mut self) -> Result<&[i32],ocl::Error>{
        match self{
            TraceProcessor::RealTracer(kernel,queue) =>{
                unpack!(kernel.execute(&queue));
                kernel.get_result(&queue)
            },
            TraceProcessor::AaTracer(kernel,clear_kernel,img_kernel,queue) =>{
                unpack!(clear_kernel.execute(&queue));
                unpack!(kernel.execute(&queue));
                unpack!(img_kernel.execute(&queue));
                img_kernel.get_result(&queue)
            },
        }
    }
}
