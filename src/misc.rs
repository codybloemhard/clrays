pub trait Incrementable{
    fn inc_pre(&mut self) -> Self;
    fn inc_post(&mut self) -> Self;
}

impl Incrementable for u32{
    fn inc_pre(&mut self) -> Self{
        *self += 1;
        *self
    }
    fn inc_post(&mut self) -> Self{
        *self += 1;
        *self - 1
    }
}

impl Incrementable for usize{
    fn inc_pre(&mut self) -> Self{
        *self += 1;
        *self
    }
    fn inc_post(&mut self) -> Self{
        *self += 1;
        *self - 1
    }
}

pub fn build_vec<T: std::default::Default + std::clone::Clone>
    (size: usize) -> Vec<T>{
        let size = size;
    let mut vec = Vec::with_capacity(size);
    let x = T::default();
    vec.resize(size, x);
    vec
}

pub fn make_nonzero_len<T: std::default::Default> (v: &mut Vec<T>){
    if v.is_empty(){
        v.push(T::default());
    }
}

pub fn load_source(path: &str) -> Result<String, std::io::Error>{
    use std::io::prelude::*;
    let mut file = std::fs::File::open(path)?;
    let mut src = String::new();
    file.read_to_string(&mut src)?;
    Ok(src)
}

#[macro_export]
macro_rules! unpackdb {
    ($x:expr, $msg:expr) =>{
        match $x{
            Ok(z) => z,
            Err(e) => return Err(format!("{}: {}", $msg, e)),
        }
    }
}

