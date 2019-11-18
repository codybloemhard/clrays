use ocl::{Buffer,flags,Queue};
use crate::misc;

pub struct ClBuffer<T: ocl::OclPrm + std::default::Default + std::clone::Clone>{
    ocl_buffer: Buffer::<T>,
    client_buffer: *mut Vec<T>,
}

impl<T: ocl::OclPrm> ClBuffer<T>{
    pub fn new(queue: &Queue, size: usize, init_val: T) -> Result<Self,ocl::Error>{
        let size = std::cmp::max(size,1);
        let ocl_buffer = match Buffer::<T>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(size)
        .fill_val(init_val)
        .build(){
            Ok(x) => x,
            Err(e) => return Err(e),
        };
        let client_buffer = &mut misc::build_vec(size);
        Ok(Self{
            ocl_buffer, client_buffer,
        })
    }

    pub fn from(queue: &Queue, vec: *mut Vec<T>) -> Result<Self,ocl::Error>{
        let ocl_buffer;
        unsafe {
            let len = vec.as_ref().unwrap().len(); 
            if len == 0 { panic!("Error: ClBuffer::from got and empty vector!"); }
            ocl_buffer = match Buffer::<T>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .use_host_slice(&*vec)
            .len(len)
            .build(){
                Ok(x) => x,
                Err(e) => return Err(e),
            };
        }
        let client_buffer = vec;
        Ok(Self{
            ocl_buffer, client_buffer,
        })
    }

    pub fn download(&mut self, queue: &Queue) -> Result<(),ocl::Error>{
        unsafe{ 
            match self.ocl_buffer.cmd()
            .queue(queue)
            .read(&mut *self.client_buffer)
            .enq(){
                Ok(_) => Ok(()),
                Err(e) => Err(e),
            }
        }
    }

    pub fn upload(&mut self, queue: &Queue) -> Result<(),ocl::Error>{
        unsafe{
            let deref = &*self.client_buffer;
            match self.ocl_buffer.cmd()
            .queue(queue)
            .write(deref)
            .enq(){
                Ok(_) => Ok(()),
                Err(e) => Err(e),
            }
        }
    }

    pub fn get_slice(&self) -> &[T]{
        unsafe { &*self.client_buffer }
    }

    pub fn get_ocl_buffer(&self) -> &Buffer<T>{
        &self.ocl_buffer
    }
}
