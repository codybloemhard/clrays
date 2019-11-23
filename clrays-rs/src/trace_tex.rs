use crate::misc::build_vec;

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
    pub fn vector_tex(path: &str) -> Result<Self,String>{
        let img = match image::open(path){
            Ok(x) => x,
            Err(e) => { println!("{}", e); return Err("could not load image.".to_string()); }
        };
        let buff = img.to_rgb();
        
        Result::Ok(Self{
            pixels: buff.to_vec(),
            width: buff.width() as i32,
            height: buff.height() as i32,
        })
    }

    pub fn scalar_tex(path: &str) -> Result<Self,String>{
        let img = match image::open(path){
            Ok(x) => x,
            Err(e) => { println!("{}", e); return Err("could not load image.".to_string()); }
        };
        let buff = img.to_rgb();
        let w = buff.width() as i32;
        let h = buff.height() as i32;
        let buff = buff.to_vec();
        let mut avg = build_vec(buff.len() / 3);
        for i in 0..avg.len(){
            let mut val = 0u16;
            val += buff[i * 3 + 0] as u16;
            val += buff[i * 3 + 1] as u16;
            val += buff[i * 3 + 2] as u16;
            val /= 3;
            avg[i] = val as u8;
        }
        Result::Ok(Self{
            pixels: avg,
            width: w,
            height: h,
        })
    }
}
