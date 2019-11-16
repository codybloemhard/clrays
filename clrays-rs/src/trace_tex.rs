pub enum TexType{
    Vector3c8bpc,
    Scalar8b,
}

pub struct TraceTex{
    pub pixels: Vec<u8>,
    pub width: i32,
    pub height: i32,
}