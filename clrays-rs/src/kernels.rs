use ocl::{Buffer,Kernel,Program,Queue};
use crate::cl_helpers::{ClBuffer};
use std::rc::Rc;

pub trait VoidKernel<T: ocl::OclPrm>{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>;
    fn get_buffer(&self) -> Rc<ClBuffer<T>>;
}

pub trait ResultKernel<T: ocl::OclPrm>{
    fn execute();
    fn get_buffer() -> Buffer<T>;
    fn get_result() -> Vec<T>;
}

pub struct ClearKernel{
    kernel: Kernel,
    buffer: Rc<ClBuffer<f32>>,
    work: (u32,u32),
}

impl ClearKernel{
    pub fn new(name: String, res:(u32,u32), program: &Program, queue: &Queue, buffer: Rc<ClBuffer::<f32>>) -> Result<Self,ocl::Error>{
        let (w,h) = res;
        let kernel = match Kernel::builder()
        .program(program)
        .name(&name)
        .queue(queue.clone())
        .global_work_size([w,h])
        .arg(buffer.get_ocl_buffer())
        .arg(&10f32)
        .build(){
            Ok(x) => x,
            Err(e) => return Err(e),
        };
        Result::Ok(Self{
            kernel,
            buffer,
            work: res,
        })
    }
}

impl VoidKernel<f32> for ClearKernel{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        unsafe { 
            match self.kernel.cmd().queue(queue).enq(){
                Ok(_) => Ok(()),
                Err(e) => Err(e),
            }
        }
    }

    fn get_buffer(&self) -> Rc<ClBuffer<f32>>{
        self.buffer.clone()
    }
}
