use ocl::{Kernel,Program,Queue};
use std::rc::Rc;
use crate::cl_helpers::{ClBuffer};
use crate::scene::Scene;
use crate::info::Info;

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
        let kernel = unpack!(Kernel::builder()
        .program(program)
        .name(name)
        .queue(queue.clone())
        .global_work_size([w,h])
        .arg(buffer.get_ocl_buffer())
        .arg(&10f32)
        .build());
        Result::Ok(Self{ kernel,buffer })
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
        let kernel = unpack!(Kernel::builder()
        .program(program)
        .name(name)
        .queue(queue.clone())
        .global_work_size([w,h])
        .arg(input.get_ocl_buffer())
        .arg(buffer.get_ocl_buffer())
        .arg(w as u32)
        .arg(h as u32)
        .build());
        Ok(Self{ buffer,kernel,dirty })
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

pub struct TraceKernel{
    kernel: Kernel,
    dirty: bool,
    buffer: ClBuffer<i32>,
    scene_params: ClBuffer<i32>,
    scene_items: ClBuffer<f32>,
    tex_params: ClBuffer<i32>,
    tex_items: ClBuffer<u8>,
    res: (u32,u32),
}

impl TraceKernel{
    pub fn new(name: &str, (w,h): (u32,u32), program: &Program, queue: &Queue, scene: &mut Scene, info: &mut Info) -> Result<Self, ocl::Error>{
        info.set_time_point("Start constructing TraceKernel");
        let dirty = false;
        let buffer = unpack!(ClBuffer::<i32>::new(queue, w as usize * h as usize, 0));
        info.set_time_point("Build int frame buffer");
        let scene_raw = scene.get_buffers();
        let scene_params_raw = scene.get_params_buffer();
        info.set_time_point("Build scene data");
        let scene_params = unpack!(ClBuffer::from(queue, scene_params_raw));
        let scene_items = unpack!(ClBuffer::from(queue, scene_raw));
        info.set_time_point("Build scene buffers");
        let tex_raw = scene.get_textures_buffer();
        let tex_params_raw = scene.get_texture_params_buffer();
        info.set_time_point("Build texture data");
        let tex_params = unpack!(ClBuffer::from(queue, tex_params_raw));
        let tex_items = unpack!(ClBuffer::from(queue, tex_raw));
        info.set_time_point("Build texture buffers");
        let kernel = unpack!(Kernel::builder()
        .program(program)
        .name(name)
        .queue(queue.clone())
        .global_work_size([w,h])
        .arg(buffer.get_ocl_buffer())
        .arg(w as u32)
        .arg(h as u32)
        .arg(scene_params.get_ocl_buffer())
        .arg(scene_items.get_ocl_buffer())
        .arg(tex_params.get_ocl_buffer())
        .arg(tex_items.get_ocl_buffer())
        .build());
        info.set_time_point("Create kernel");
        Ok(Self{ kernel, dirty, buffer, scene_params, scene_items, tex_params, tex_items, res: (w,h) })
    }

    /*pub fn update(&mut self, queue: &Queue) -> Result<(),ocl::Error>{
        match self.scene_params.upload(queue){
            Ok(_) => {}, Err(e) => return Err(e),
        }
        match self.scene_items.upload(queue){
            Ok(_) => {}, Err(e) => return Err(e),
        }
        match self.tex_params.upload(queue){
            Ok(_) => {}, Err(e) => return Err(e),
        }
        match self.tex_items.upload(queue){
            Ok(_) => {}, Err(e) => return Err(e),
        }
        Ok(())
    }*/

    pub fn get_res(&self) -> (u32,u32){
        self.res
    }
}

impl VoidKernel<i32> for TraceKernel{
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

impl ResultKernel<i32> for TraceKernel{
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
