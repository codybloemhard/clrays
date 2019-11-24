use ocl::{Queue,Program};
use crate::kernels::*;
use crate::scene::{Scene};
use crate::info::{Info};
use crate::cl_helpers::{create_five};
use crate::misc::{load_source};

#[derive(Clone,Copy,PartialEq)]
pub enum TraceType{
    Real,
    AA,
}

pub struct TraceProcessor{
    trace_real_kernel: TraceKernel,
    ttype: TraceType,
    queue: Queue,
}

impl TraceProcessor{
    pub fn new((width,height): (u32,u32), aa: u32, scene: &mut Scene, info: &mut Info, ttype: TraceType) -> Result<Self,String>{
        info.start_time();
        let src = unpackdb!(load_source("../Assets/Kernels/raytrace.cl"));
        info.set_time_point("Loading source file");
        let (_,_,_,program,queue) = unpackdb!(create_five(&src));
        info.set_time_point("Creating OpenCL objects");
        let kernel = unpackdb!(TraceKernel::new("raytracing", (width,height), &program, &queue, scene, info));
        info.set_time_point("Last time stamp");
        info.stop_time();
        info.print_info();
        Ok(Self{
            trace_real_kernel: kernel,
            ttype,
            queue
        })
    }

    pub fn render(&mut self) -> Result<&[i32],ocl::Error>{
        unpack!(self.trace_real_kernel.execute(&self.queue));
        self.trace_real_kernel.get_result(&self.queue)
    }
}
