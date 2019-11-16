pub trait Incrementable{
    fn inc(&mut self) -> Self;
}

impl Incrementable for i32{
    fn inc(&mut self) -> Self{
        *self += 1;
        *self
    }
}