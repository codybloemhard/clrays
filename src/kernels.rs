use crate::cl_helpers::{ ClBufferRW, ClBufferR };
use crate::scene::Scene;
use crate::info::Info;
use crate::state::State;

use ocl::{ Kernel, Program, Queue };

pub trait VoidKernel{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>;
}

pub trait BufferKernel<T: ocl::OclPrm>{
    fn get_buffer(&self) -> &ClBufferRW<T>;
}

pub trait ResultKernel<T: ocl::OclPrm>{
    fn get_result(&mut self, queue: &Queue) -> Result<&[T], ocl::Error>;
}

pub struct ClearKernel{
    kernel: Kernel,
}

impl ClearKernel{
    pub fn new(name: &str, (w, h): (u32, u32), program: &Program, queue: &Queue, buffer: &ClBufferRW<f32>) -> Result<Self, ocl::Error>{
        let kernel = Kernel::builder()
            .program(program)
            .name(name)
            .queue(queue.clone())
            .global_work_size([w, h])
            .arg(buffer.get_ocl_buffer())
            .arg(w)
            .build()?;
        Result::Ok(Self{ kernel })
    }
}

impl VoidKernel for ClearKernel{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        unsafe {
            self.kernel.cmd().queue(queue).enq().map(|_| ())
        }
    }
}

pub struct ImageKernel{
    kernel: Kernel,
    buffer: ClBufferRW<u32>,
    dirty: bool,
}

impl ImageKernel{
    pub fn new(name: &str, (w, h): (u32, u32), tm: u32, program: &Program, queue: &Queue, input: &ClBufferRW<f32>) -> Result<Self, ocl::Error>{
        let dirty = false;
        let buffer = ClBufferRW::<u32>::new(queue, w as usize * h as usize, 0)?;
        let kernel = Kernel::builder()
            .program(program)
            .name(name)
            .queue(queue.clone())
            .global_work_size([w, h])
            .arg(input.get_ocl_buffer())
            .arg(buffer.get_ocl_buffer())
            .arg(w as u32)
            .arg(1.0f32) // scaling
            .arg(tm) // tonemap
            .build()?;
        Ok(Self{ buffer, kernel, dirty })
    }

    pub fn set_divider(&mut self, div: f32) -> Result<(), ocl::Error>{
        self.kernel.set_arg(3, 1.0 / div)?;
        Ok(())
    }
}

impl VoidKernel for ImageKernel{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        unsafe {
            self.kernel.cmd().queue(queue).enq().map(|_| self.dirty = true)
        }
    }
}

impl BufferKernel<u32> for ImageKernel{
    fn get_buffer(&self) -> &ClBufferRW<u32>{
        &self.buffer
    }
}

impl ResultKernel<u32> for ImageKernel{
    fn get_result(&mut self, queue: &Queue) -> Result<&[u32], ocl::Error>{
        if self.dirty {
            self.buffer.download(queue)?;
        }
        self.dirty = false;
        Ok(self.buffer.get_slice())
    }
}

// Wrong name for this struct breaks clippy/compiler
// If you name it back TraceKernelReal it's fine
// Crashes with `cargo clippy --all-features` and not with `cargo check`
pub struct TraceKernelWhitted{
    kernel: Kernel,
    dirty: bool,
    buffer: ClBufferRW<u32>,
    scene_params: ClBufferRW<u32>,
}

impl TraceKernelWhitted{
    pub fn new(name: &str, (w, h): (u32, u32), program: &Program, queue: &Queue, scene: &mut Scene, info: &mut Info) -> Result<Self, ocl::Error>{
        info.set_time_point("Start constructing kernel");
        let dirty = false;
        let buffer = ClBufferRW::<u32>::new(queue, w as usize * h as usize, 0)?;
        info.int_buffer_size = w as u64 * h as u64;
        info.set_time_point("Build int frame buffer");
        let scene_params_raw = scene.get_scene_params_buffer();
        let scene_raw = scene.get_scene_buffer();
        let bvh_raw = scene.get_bvh_buffer();
        info.scene_size = scene_raw.len() as u64;
        info.meta_size = scene_params_raw.len() as u64;
        info.bvh_size = bvh_raw.len() as u64;
        info.set_time_point("Build scene data");
        let mut scene_params = ClBufferRW::from(queue, scene_params_raw)?;
        let mut scene_items = ClBufferR::new(queue, scene_raw.len(), 0.0)?;
        let mut bvh = ClBufferR::new(queue, bvh_raw.len(), 0)?;
        info.set_time_point("Build scene buffers");
        let tex_raw = scene.get_textures_buffer();
        let tex_params_raw = scene.get_texture_params_buffer();
        info.set_time_point("Build texture data");
        let mut tex_params = ClBufferR::new(queue, tex_params_raw.len(), 0)?;
        let mut tex_items = ClBufferR::new(queue, tex_raw.len(), 0)?;
        info.set_time_point("Build texture buffers");
        let mut kbuilder = Kernel::builder();
        kbuilder.program(program);
        kbuilder.name(name);
        kbuilder.queue(queue.clone());
        kbuilder.global_work_size([w, h]);
        kbuilder.arg(buffer.get_ocl_buffer());
        kbuilder.arg(w as u32);
        kbuilder.arg(h as u32);
        kbuilder.arg(scene_params.get_ocl_buffer());
        kbuilder.arg(scene_items.get_ocl_buffer());
        kbuilder.arg(bvh.get_ocl_buffer());
        kbuilder.arg(tex_params.get_ocl_buffer());
        kbuilder.arg(tex_items.get_ocl_buffer());
        let kernel = kbuilder.build()?;
        info.set_time_point("Create kernel");
        /*Choice: Either: upload an let ClBufferRW's go out of scope
        Or: store ClBufferRW's in the struct so they still live.
        They will be uploaded automatically then.
        I choose to upload here and let them go, as i don't need them later on and i can time the uploading.
        Except the scene_params. It is small and used to change camera etc. */
        scene_params.upload(queue)?;
        info.set_time_point("Upload scene parameters");
        scene_items.upload_new(queue, &scene_raw)?;
        info.set_time_point("Upload scene items");
        bvh.upload_new(queue, &bvh_raw)?;
        info.set_time_point("Upload bvh");
        tex_params.upload_new(queue, &tex_params_raw)?;
        info.set_time_point("Upload texture parameters");
        tex_items.upload_new(queue, &tex_raw)?;
        info.set_time_point("Upload textures");
        Ok(Self{ kernel, dirty, buffer, scene_params })
    }

    pub fn update(&mut self, queue: &Queue, scene: &mut Scene) -> Result<(), ocl::Error>{
        let scene_params_raw = scene.get_scene_params_buffer();
        self.scene_params.upload_new(queue, &scene_params_raw)?;
        Ok(())
    }
}

impl VoidKernel for TraceKernelWhitted{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        unsafe {
            self.kernel.cmd().queue(queue).enq().map(|_| self.dirty = true)
        }
    }
}

impl BufferKernel<u32> for TraceKernelWhitted{
    fn get_buffer(&self) -> &ClBufferRW<u32>{
        &self.buffer
    }
}

impl ResultKernel<u32> for TraceKernelWhitted{
    fn get_result(&mut self, queue: &Queue) -> Result<&[u32], ocl::Error>{
        if self.dirty {
            self.buffer.download(queue)?;
        }
        self.dirty = false;
        Ok(self.buffer.get_slice())
    }
}

pub struct TraceKernelPath{
    kernel: Kernel,
    buffer: ClBufferRW<f32>,
    scene_params: ClBufferRW<u32>,
}

impl TraceKernelPath{
    pub fn new(name: &str, (w, h): (u32, u32), program: &Program, queue: &Queue, scene: &mut Scene, info: &mut Info) -> Result<Self, ocl::Error>{
        info.set_time_point("Start constructing kernel");
        let bsize = w as usize * h as usize * 3;
        let buffer = ClBufferRW::<f32>::new(queue, bsize, 0.0)?;
        info.float_buffer_size = bsize as u64;
        info.set_time_point("Build float frame buffer");
        let scene_params_raw = scene.get_scene_params_buffer();
        let scene_raw = scene.get_scene_buffer();
        let bvh_raw = scene.get_bvh_buffer();
        info.scene_size = scene_raw.len() as u64;
        info.meta_size = scene_params_raw.len() as u64;
        info.bvh_size = bvh_raw.len() as u64;
        info.set_time_point("Build scene data");
        let mut scene_params = ClBufferRW::from(queue, scene_params_raw)?;
        let mut scene_items = ClBufferR::new(queue, scene_raw.len(), 0.0)?;
        let mut bvh = ClBufferR::new(queue, bvh_raw.len(), 0)?;
        info.set_time_point("Build scene buffers");
        let tex_raw = scene.get_textures_buffer();
        let tex_params_raw = scene.get_texture_params_buffer();
        info.set_time_point("Build texture data");
        let mut tex_params = ClBufferR::new(queue, tex_params_raw.len(), 0)?;
        let mut tex_items = ClBufferR::new(queue, tex_raw.len(), 0)?;
        info.set_time_point("Build texture buffers");
        let mut kbuilder = Kernel::builder();
        kbuilder.program(program);
        kbuilder.name(name);
        kbuilder.queue(queue.clone());
        kbuilder.global_work_size([w, h]);
        kbuilder.arg(buffer.get_ocl_buffer());
        kbuilder.arg(w as u32);
        kbuilder.arg(h as u32);
        kbuilder.arg(0u32);
        kbuilder.arg(scene_params.get_ocl_buffer());
        kbuilder.arg(scene_items.get_ocl_buffer());
        kbuilder.arg(bvh.get_ocl_buffer());
        kbuilder.arg(tex_params.get_ocl_buffer());
        kbuilder.arg(tex_items.get_ocl_buffer());
        let kernel = kbuilder.build()?;
        info.set_time_point("Create kernel");
        scene_params.upload(queue)?;
        info.set_time_point("Upload scene parameters");
        scene_items.upload_new(queue, &scene_raw)?;
        info.set_time_point("Upload scene items");
        bvh.upload_new(queue, &bvh_raw)?;
        info.set_time_point("Upload bvh");
        tex_params.upload_new(queue, &tex_params_raw)?;
        info.set_time_point("Upload texture parameters");
        tex_items.upload_new(queue, &tex_raw)?;
        info.set_time_point("Upload textures");
        Ok(Self{ kernel, buffer, scene_params })
    }

    pub fn update(&mut self, queue: &Queue, scene: &mut Scene, state: &State) -> Result<(), ocl::Error>{
        let scene_params_raw = scene.get_scene_params_buffer();
        self.scene_params.upload_new(queue, &scene_params_raw)?;
        self.kernel.set_arg(3, state.samples_taken as u32)?;
        Ok(())
    }

    pub fn frame_energy(&mut self, queue: &Queue) -> f32{
        self.buffer.download(queue).expect("Could not download buffer to calculate frame energy!");
        let mut e = 0.0;
        for f in self.buffer.get_slice(){
            e += *f;
        }
        e
    }
}

impl VoidKernel for TraceKernelPath{
    fn execute(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        unsafe {
            self.kernel.cmd().queue(queue).enq()
        }
    }
}

impl BufferKernel<f32> for TraceKernelPath{
    fn get_buffer(&self) -> &ClBufferRW<f32>{
        &self.buffer
    }
}

