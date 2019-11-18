use ocl::{Buffer,flags,Queue};

pub struct ClBuffer<T: ocl::OclPrm>{
    ocl_buffer: Buffer::<T>,
    client_buffer: *mut Vec<T>,
}

impl<T: ocl::OclPrm> ClBuffer<T>{
    pub fn new(queue: &Queue, size: usize, init_val: T) -> Result<Self,ocl::Error>{
        let ocl_buffer = match Buffer::<T>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .len(size)
        .fill_val(init_val)
        .build(){
            Ok(x) => x,
            Err(e) => return Err(e),
        };
        let client_buffer = &mut Vec::with_capacity(size);
        Ok(Self{
            ocl_buffer, client_buffer,
        })
    }

    pub fn from(queue: &Queue, vec: *mut Vec<T>) -> Result<Self,ocl::Error>{
        let ocl_buffer;
        unsafe {
            ocl_buffer = match Buffer::<T>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_WRITE)
            .use_host_slice(&*vec)
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
