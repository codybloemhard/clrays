use ocl::{Kernel,Program,Queue};
use ocl::builders::{KernelBuilder};
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
        .arg(w)
        .arg(h)
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
    pub fn new(name: &str, (w,h): (u32,u32), program: &Program, queue: &Queue, input: &ClBuffer<f32>) -> Result<Self,ocl::Error>{
        let dirty = false;
        let buffer = match ClBuffer::<i32>::new(queue, w as usize * h as usize, 0){
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

pub struct TraceKernelReal{
    kernel: Kernel,
    dirty: bool,
    buffer: ClBuffer<i32>,
    scene_params: ClBuffer<i32>,
    res: (u32,u32),
}

impl TraceKernelReal{
    pub fn new(name: &str, (w,h): (u32,u32), program: &Program, queue: &Queue, scene: &mut Scene, info: &mut Info) -> Result<Self, ocl::Error>{
        info.set_time_point("Start constructing kernel");
        let dirty = false;
        let buffer = unpack!(ClBuffer::<i32>::new(queue, w as usize * h as usize, 0));
        info.int_buffer_size = w as u64 * h as u64;
        info.set_time_point("Build int frame buffer");
        let scene_params_raw = scene.get_params_buffer();
        let scene_raw = scene.get_buffers();
        info.scene_size = scene_raw.len() as u64;
        info.meta_size = scene_params_raw.len() as u64;
        info.set_time_point("Build scene data");
        let mut scene_params = unpack!(ClBuffer::from(queue, scene_params_raw));
        let mut scene_items = unpack!(ClBuffer::from(queue, scene_raw));
        info.set_time_point("Build scene buffers");
        let tex_raw = scene.get_textures_buffer();
        let tex_params_raw = scene.get_texture_params_buffer();
        info.set_time_point("Build texture data");
        let mut tex_params = unpack!(ClBuffer::from(queue, tex_params_raw));
        let mut tex_items = unpack!(ClBuffer::from(queue, tex_raw));
        info.set_time_point("Build texture buffers");
        let mut kbuilder = Kernel::builder();
        kbuilder.program(program);
        kbuilder.name(name);
        kbuilder.queue(queue.clone());
        kbuilder.global_work_size([w,h]);

        kbuilder.arg(buffer.get_ocl_buffer());
        kbuilder.arg(w as u32);
        kbuilder.arg(h as u32);

        kbuilder.arg(scene_params.get_ocl_buffer());
        kbuilder.arg(scene_items.get_ocl_buffer());
        kbuilder.arg(tex_params.get_ocl_buffer());
        kbuilder.arg(tex_items.get_ocl_buffer());
        let kernel = unpack!(kbuilder.build());
        info.set_time_point("Create kernel");
        /*Choice: Either: upload an let ClBuffer's go out of scope
        Or: store ClBuffer's in the struct so they still live.
        They will be uploaded automatically then.
        I choose to upload here and let them go, as i don't need them later on and i can time the uploading.
        Except tehe scene_params. It is small and used to change camera etc. */
        unpack!(scene_params.upload(&queue));
        info.set_time_point("Upload scene_params");
        unpack!(scene_items.upload(&queue));
        info.set_time_point("Upload scene_items");
        unpack!(tex_params.upload(&queue));
        info.set_time_point("Upload tex_params");
        unpack!(tex_items.upload(&queue));
        info.set_time_point("Upload tex_items");
        Ok(Self{ kernel, dirty, buffer, scene_params, res: (w,h) })
    }

    pub fn update(&mut self, queue: &Queue) -> Result<(),ocl::Error>{
        unpack!(self.scene_params.upload(queue));
        Ok(())
    }

    pub fn get_res(&self) -> (u32,u32){
        self.res
    }
}

impl VoidKernel<i32> for TraceKernelReal{
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

impl ResultKernel<i32> for TraceKernelReal{
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

pub struct TraceKernelAa{
    kernel: Kernel,
    dirty: bool,
    buffer: Rc<ClBuffer<f32>>,
    scene_params: ClBuffer<i32>,
    res: (u32,u32),
}

impl TraceKernelAa{
    pub fn new(name: &str, (w,h): (u32,u32), aa: u32, program: &Program, queue: &Queue, scene: &mut Scene, info: &mut Info) -> Result<Self, ocl::Error>{
        info.set_time_point("Start constructing kernel");
        let dirty = false;
        let bsize = w as usize * h as usize * 3;
        let buffer = unpack!(ClBuffer::<f32>::new(queue, bsize, 0.0));
        info.float_buffer_size = bsize as u64;
        info.set_time_point("Build float frame buffer");
        let scene_params_raw = scene.get_params_buffer();
        let scene_raw = scene.get_buffers();
        info.scene_size = scene_raw.len() as u64;
        info.meta_size = scene_params_raw.len() as u64;
        info.set_time_point("Build scene data");
        let mut scene_params = unpack!(ClBuffer::from(queue, scene_params_raw));
        let mut scene_items = unpack!(ClBuffer::from(queue, scene_raw));
        info.set_time_point("Build scene buffers");
        let tex_raw = scene.get_textures_buffer();
        let tex_params_raw = scene.get_texture_params_buffer();
        info.set_time_point("Build texture data");
        let mut tex_params = unpack!(ClBuffer::from(queue, tex_params_raw));
        let mut tex_items = unpack!(ClBuffer::from(queue, tex_raw));
        info.set_time_point("Build texture buffers");
        let mut kbuilder = Kernel::builder();
        kbuilder.program(program);
        kbuilder.name(name);
        kbuilder.queue(queue.clone());
        kbuilder.global_work_size([w * aa, h * aa]);
        kbuilder.arg(buffer.get_ocl_buffer());
        kbuilder.arg(w as u32);
        kbuilder.arg(h as u32);
        kbuilder.arg(aa);
        kbuilder.arg(scene_params.get_ocl_buffer());
        kbuilder.arg(scene_items.get_ocl_buffer());
        kbuilder.arg(tex_params.get_ocl_buffer());
        kbuilder.arg(tex_items.get_ocl_buffer());
        let kernel = unpack!(kbuilder.build());
        info.set_time_point("Create kernel");
        unpack!(scene_params.upload(&queue));
        info.set_time_point("Upload scene_params");
        unpack!(scene_items.upload(&queue));
        info.set_time_point("Upload scene_items");
        unpack!(tex_params.upload(&queue));
        info.set_time_point("Upload tex_params");
        unpack!(tex_items.upload(&queue));
        info.set_time_point("Upload tex_items");
        Ok(Self{ kernel, dirty, buffer: Rc::new(buffer), scene_params, res: (w,h) })
    }

    pub fn update(&mut self, queue: &Queue) -> Result<(),ocl::Error>{
        unpack!(self.scene_params.upload(queue));
        Ok(())
    }

    pub fn get_res(&self) -> (u32,u32){
        self.res
    }

    pub fn get_buffer_rc(&self) -> Rc<ClBuffer<f32>>{
        self.buffer.clone()
    }
}

impl VoidKernel<f32> for TraceKernelAa{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        unsafe {
            match self.kernel.cmd().queue(queue).enq(){
                Ok(_) => { self.dirty = true; Ok(()) },
                Err(e) => Err(e),
            }
        }
    }

    fn get_buffer(&self) -> &ClBuffer<f32>{
        &self.buffer
    }
}

/*impl ResultKernel<f32> for TraceKernelAa{
    fn get_result(&mut self, queue: &Queue) -> Result<&[f32],ocl::Error>{
        if self.dirty {
            match self.buffer.download(queue){
                Ok(_) => {},
                Err(e) => return Err(e),
            }
        }
        self.dirty = false;
        Ok(self.buffer.get_slice())
    }
}*/
