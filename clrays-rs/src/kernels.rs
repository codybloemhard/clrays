use ocl::{Buffer,Kernel,Program,Queue};
use crate::cl_helpers::{ClBuffer};
use std::rc::Rc;

pub trait VoidKernel<T: ocl::OclPrm>{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>;
    fn get_buffer(&self) -> &ClBuffer<T>;
}

pub trait ResultKernel<T: ocl::OclPrm>{
    fn get_result(&mut self, queue: &Queue) -> Result<&[T],ocl::Error>;
}

pub struct ClearKernel{
    kernel: Kernel,
    buffer: Rc<ClBuffer<f32>>,
}

impl ClearKernel{
    pub fn new(name: &str, (w,h): (u32,u32), program: &Program, queue: &Queue, buffer: Rc<ClBuffer<f32>>) -> Result<Self,ocl::Error>{
        let kernel = match Kernel::builder()
        .program(program)
        .name(name)
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

    fn get_buffer(&self) -> &ClBuffer<f32>{
        &self.buffer
    }
}

pub struct ImageKernel{
    kernel: Kernel,
    buffer: ClBuffer<i32>,
    dirty: bool,
}

impl ImageKernel{
    pub fn new(name: &str, (w,h): (usize,usize), program: &Program, queue: &Queue, input: &ClBuffer<f32>) -> Result<Self,ocl::Error>{
        let dirty = false;
        let buffer = match ClBuffer::<i32>::new(queue, w * h, 0){
            Ok(x) => x,
            Err(e) => return Err(e),
        };
        let kernel = match Kernel::builder()
        .program(program)
        .name(name)
        .queue(queue.clone())
        .global_work_size([w,h])
        .arg(input.get_ocl_buffer())
        .arg(buffer.get_ocl_buffer())
        .arg(w as u32)
        .arg(h as u32)
        .build(){
            Ok(x) => x,
            Err(e) => return Err(e),
        };
        Ok(Self{
            buffer,
            kernel,
            dirty,
        })
    }
}

impl VoidKernel<i32> for ImageKernel{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        unsafe { 
            match self.kernel.cmd().queue(queue).enq(){
                Ok(_) => { self.dirty = true; Ok(()) },
                Err(e) => Err(e),
            }
        }
    }

    fn get_buffer(&self) -> &ClBuffer<i32>{
        &self.buffer
    }
}

impl ResultKernel<i32> for ImageKernel{
    fn get_result(&mut self, queue: &Queue) -> Result<&[i32],ocl::Error>{
        if self.dirty {
            match self.buffer.download(queue){
                Ok(_) => {},
                Err(e) => return Err(e),
            }
        }
        self.dirty = false;
        Ok(self.buffer.get_slice())
    }
}


