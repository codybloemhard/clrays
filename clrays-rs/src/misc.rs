pub trait Incrementable{
    fn inc_pre(&mut self) -> Self;
    fn inc_post(&mut self) -> Self;
}

impl Incrementable for i32{
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
    if v.len() == 0{
        v.push(T::default());
    }
}

pub fn load_source(path: &str) -> Result<String,std::io::Error>{
    use std::io::prelude::*;
    let mut file = match std::fs::File::open(path){
        Ok(x) => x,
        Err(e) => return Err(e),
    };
    let mut src = String::new();
    let err = file.read_to_string(&mut src);
    if let Err(e) = err{
        return Err(e);
    }
    Ok(src)
}

#[macro_export]
macro_rules! unpack {
    ($x:expr) =>{
        match $x{
            Ok(z) => z,
            Err(e) => return Err(format!("{:?}", e)),
        };
    }
}
