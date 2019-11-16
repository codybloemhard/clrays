#[derive(PartialEq,Copy,Clone)]
pub enum TexType{
    Vector3c8bpc,
    Scalar8b,
}

pub struct TraceTex{
    pub pixels: Vec<u8>,
    pub width: i32,
    pub height: i32,
}

impl TraceTex{
    pub fn vector_tex(path: &str) -> Self{//TODO: implement
        Self{
            pixels: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    pub fn scalar_tex(path: &str) -> Self{//TODO: implement
        Self{
            pixels: Vec::new(),
            width: 0,
            height: 0,
        }
    }
}