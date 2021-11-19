use crate::misc;

use ocl::{ Buffer, flags, Queue, Program, Device, Platform, Context};

pub struct ClBuffer<T: ocl::OclPrm + std::default::Default + std::clone::Clone>{
    ocl_buffer: Buffer::<T>,
    client_buffer: Vec<T>,
}

impl<T: ocl::OclPrm> ClBuffer<T>{
    pub fn new(queue: &Queue, size: usize, init_val: T) -> Result<Self, ocl::Error>{
        let size = std::cmp::max(size, 1);
        let ocl_buffer = Buffer::<T>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .len(size)
            .fill_val(init_val)
            .build()?;
        let client_buffer = misc::build_vec(size);
        Ok(Self{
            ocl_buffer, client_buffer,
        })
    }

    pub fn from(queue: &Queue, vec: Vec<T>) -> Result<Self, String>{
        let ocl_buffer;
        unsafe {
            let len = vec.len();
            if len == 0 { return Err("ClBuffer::from got and empty vector!".to_string()); }
            ocl_buffer = unpackdb!(Buffer::<T>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .use_host_slice(&vec)
                .len(len)
                .build(), "Could not build ocl buffer!");
        }
        let client_buffer = vec;
        Ok(Self{
            ocl_buffer, client_buffer,
        })
    }

    pub fn download(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        self.ocl_buffer.cmd()
            .queue(queue)
            .read(&mut self.client_buffer)
            .enq()
    }

    pub fn upload(&mut self, queue: &Queue) -> Result<(), ocl::Error>{
        self.ocl_buffer.cmd()
            .queue(queue)
            .write(&self.client_buffer)
            .enq()
    }

    pub fn get_slice(&self) -> &[T]{
        &self.client_buffer
    }

    pub fn get_ocl_buffer(&self) -> &Buffer<T>{
        &self.ocl_buffer
    }
}

pub fn create_five(src: &str) -> Result<(Platform, Device, Context, Program, Queue), ocl::Error>{
    let platform = Platform::default();
    let device = Device::first(platform)?;
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let program = Program::builder()
        .devices(device)
        .src(src)
        .build(&context)?;
    let queue = Queue::new(&context, device, None)?;
    Ok((platform, device, context, program, queue))
}
