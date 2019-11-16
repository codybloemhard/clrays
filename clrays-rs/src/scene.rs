use crate::vec3::Vec3;
use crate::trace_tex::{TexType, TraceTex};
use std::collections::HashMap;

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

pub struct Scene{
    pub spheres: Vec<Sphere>,
    pub planes: Vec<Plane>,
    pub boxes: Vec<Plane>,
    pub lights: Vec<Light>,
    scene_params: [i32; Self::SCENE_PARAM_SIZE],
    next_texture: i32,
    ghost_textures: HashMap<String, (String, TexType)>,
    textures_ids: HashMap<String, i32>,
    textures: Vec<TraceTex>,
    skybox: i32,
    pub sky_col: Vec3,
    pub sky_intensity: f32,
    pub cam_pos: Vec3,
    pub cam_dir: Vec3,
}

impl Scene{
    const SCENE_SIZE: i32 = 11;
    const SCENE_PARAM_SIZE: usize = 4 * 3 + Self::SCENE_SIZE as usize;
    const MATERIAL_SIZE: i32 = 10;
    const LIGHT_SIZE: i32 = 7;
    const SPHERE_SIZE: i32 = 4 + Self::MATERIAL_SIZE;
    const PLANE_SIZE: i32 = 6 + Self::MATERIAL_SIZE;
    const BOX_SIZE: i32 = 6 + Self::MATERIAL_SIZE;

    pub fn new() -> Self{
        Self{
            spheres: Vec::new(),
            planes: Vec::new(),
            boxes: Vec::new(),
            lights: Vec::new(),
            scene_params: [0; Self::SCENE_PARAM_SIZE],
            next_texture: 0,
            ghost_textures: HashMap::new(),
            textures_ids: HashMap::new(),
            textures: Vec::new(),
            skybox: 0,
            sky_col: Vec3::one(),
            sky_intensity: 1.0,
            cam_pos: Vec3::zero(),
            cam_dir: Vec3::backward(),
        }
    }

    pub fn update(&mut self){
        self.get_params_buffer();
    }

    pub fn get_params_buffer(&mut self){
        let mut i = 0;
        self.scene_params[0] = Self::LIGHT_SIZE;
        self.scene_params[1] = self.lights.len() as i32;
        self.scene_params[2] = i; i += self.lights.len() as i32 * Self::LIGHT_SIZE;
        self.scene_params[3] = Self::SPHERE_SIZE;
        self.scene_params[4] = self.spheres.len() as i32;
        self.scene_params[5] = i; i += self.spheres.len() as i32 * Self::SPHERE_SIZE;
        self.scene_params[6] = Self::PLANE_SIZE;
        self.scene_params[7] = self.planes.len() as i32;
        self.scene_params[8] = i; i += self.planes.len() as i32 * Self::PLANE_SIZE;
        self.scene_params[9] = Self::BOX_SIZE;
        self.scene_params[10] = self.boxes.len() as i32;
        self.scene_params[11] = i; //i += self.boxes.len() as i32 * Self::BOX_SIZE;
        //scene
        self.scene_params[12] = self.skybox;
        self.put_in_scene_params(13, self.sky_col);
        self.scene_params[16] = Self::f32_transm_i32(self.sky_intensity);
        self.put_in_scene_params(17, self.cam_pos);
        self.put_in_scene_params(20, self.cam_dir);
    }

    fn f32_transm_i32(f: f32) -> i32{
        unsafe { std::mem::transmute(f) }
    }

    fn put_in_scene_params(&mut self, i: usize, v: Vec3){
        self.scene_params[i + 0] = Self::f32_transm_i32(v.x);
        self.scene_params[i + 1] = Self::f32_transm_i32(v.y);
        self.scene_params[i + 2] = Self::f32_transm_i32(v.z);
    }
}