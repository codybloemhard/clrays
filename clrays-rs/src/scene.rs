use crate::vec3::Vec3;

pub struct Material{
    pub col: Vec3,
    pub reflectivity: f32,
    pub roughness: f32,
    pub texture: i32,
    pub normal_map: i32,
    pub roughness_map: i32,
    pub metalic_map: i32,
    tex_scale: f32,
}

impl Material{
    pub fn get_tex_scale(&self) -> f32{
        self.tex_scale
    }

    pub fn set_tex_scale(&mut self, v: f32){
        self.tex_scale = 1.0 / v;
    }

    pub fn basic() -> Self{
        Self{
            col: Vec3::one(),
            reflectivity: 1.0,
            roughness: 1.0,
            texture: 0,
            normal_map: 0,
            roughness_map: 0,
            metalic_map: 0,
            tex_scale: 1.0,
        }
    }
}

pub trait SceneItem{
    fn get_pos(&self) -> Vec3;
    fn get_data(&self) -> Vec<f32>;
}

pub struct Plane{
    pub pos: Vec3,
    pub nor: Vec3,
    pub mat: Material,
}

impl SceneItem for Plane{
    fn get_pos(&self) -> Vec3{
        self.pos
    }

    fn get_data(&self) -> Vec<f32>{
        vec![self.pos.x, self.pos.y, self.pos.z,
        self.nor.x, self.nor.y, self.nor.z,
        self.mat.col.x, self.mat.col.y, self.mat.col.z,
        self.mat.reflectivity, self.mat.roughness,
        self.mat.texture as f32, self.mat.normal_map as f32,
        self.mat.roughness_map as f32, self.mat.metalic_map as f32,
        self.mat.tex_scale]
    }
}

pub struct Sphere{
    pub pos: Vec3,
    pub rad: f32,
    pub mat: Material,
}

impl SceneItem for Sphere{
    fn get_pos(&self) -> Vec3{
        self.pos
    }

    fn get_data(&self) -> Vec<f32>{
        vec![self.pos.x, self.pos.y, self.pos.z, self.rad,
        self.mat.col.x, self.mat.col.y, self.mat.col.z,
        self.mat.reflectivity, self.mat.roughness,
        self.mat.texture as f32, self.mat.normal_map as f32,
        self.mat.roughness_map as f32, self.mat.metalic_map as f32,
        self.mat.tex_scale]
    }
}

pub struct Box{
    pub pos: Vec3,
    pub size: Vec3,
    pub mat: Material,
}

impl SceneItem for Box{
    fn get_pos(&self) -> Vec3{
        self.pos
    }

    fn get_data(&self) -> Vec<f32>{
        let hs = self.size.scaled(0.5);
        vec![self.pos.x - hs.x, self.pos.y - hs.y, self.pos.z - hs.z,
        self.pos.x + hs.x, self.pos.y + hs.y, self.pos.z + hs.z,
        self.mat.texture as f32, self.mat.normal_map as f32,
        self.mat.roughness_map as f32, self.mat.metalic_map as f32,
        self.mat.tex_scale]
    }
}

pub struct Light{
    pub pos: Vec3,
    pub intensity: f32,
    pub col: Vec3,
}

impl SceneItem for Light{
    fn get_pos(&self) -> Vec3{
        self.pos
    }

    fn get_data(&self) -> Vec<f32>{
        vec![self.pos.x, self.pos.y, self.pos.z,
        self.intensity, self.col.x, self.col.y, self.col.z]
    }
}
