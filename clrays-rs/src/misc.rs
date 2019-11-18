pub trait Incrementable{
    fn inc(&mut self) -> Self;
}

impl Incrementable for i32{
    fn inc(&mut self) -> Self{
        *self += 1;
        *self
    }
}

pub fn build_vec<T: std::default::Default + std::clone::Clone>
    (size: usize) -> Vec<T>{
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