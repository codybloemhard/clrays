use crate::vec3::Vec3;
use crate::trace_tex::{TexType, TraceTex};
use crate::misc::{Incrementable,build_vec,make_nonzero_len};
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
            reflectivity: 0.0,
            roughness: 1.0,
            texture: 0,
            normal_map: 0,
            roughness_map: 0,
            metalic_map: 0,
            tex_scale: 1.0,
        }
    }

    pub fn with_colour(mut self, col: Vec3) -> Self{
        self.col = col;
        self
    }

    pub fn with_reflectivity(mut self, refl: f32) -> Self{
        self.reflectivity = refl;
        self
    }

    pub fn with_roughness(mut self, roughn: f32) -> Self{
        self.roughness = roughn;
        self
    }

    pub fn with_texture(mut self, tex: i32) -> Self{
        self.texture = tex;
        self
    }

    pub fn with_normal_map(mut self, norm: i32) -> Self{
        self.normal_map = norm;
        self
    }

    pub fn with_roughness_map(mut self, roughm: i32) -> Self{
        self.roughness_map = roughm;
        self
    }

    pub fn with_metalic_map(mut self, metalm: i32) -> Self{
        self.metalic_map = metalm;
        self
    }

    pub fn with_tex_scale(mut self, v: f32) -> Self{
        self.tex_scale = 1.0 / v;
        self
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
    pub boxes: Vec<Box>,
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
            sky_intensity: 0.0,
            cam_pos: Vec3::zero(),
            cam_dir: Vec3::backward(),
        }
    }

    pub fn get_buffers(&mut self) -> Vec<f32>{
        let mut len = self.lights.len() * Self::LIGHT_SIZE as usize;
        len += self.spheres.len() * Self::SPHERE_SIZE as usize;
        len += self.planes.len() * Self::PLANE_SIZE as usize;
        len += self.boxes.len() * Self::BOX_SIZE as usize;
        let mut res = build_vec(len);
        let mut i = 0;
        Self::bufferize(&mut res, &mut i, &self.lights, Self::LIGHT_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.spheres, Self::SPHERE_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.planes, Self::PLANE_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.boxes, Self::BOX_SIZE as usize);
        make_nonzero_len(&mut res);
        res
    }

    pub fn get_params_buffer(&mut self) -> Vec<i32>{
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
        self.scene_params.to_vec()
    }

    fn f32_transm_i32(f: f32) -> i32{
        unsafe { std::mem::transmute(f) }
    }

    fn put_in_scene_params(&mut self, i: usize, v: Vec3){
        self.scene_params[i + 0] = Self::f32_transm_i32(v.x);
        self.scene_params[i + 1] = Self::f32_transm_i32(v.y);
        self.scene_params[i + 2] = Self::f32_transm_i32(v.z);
    }

    pub fn get_textures_buffer(&self) -> Vec<u8>{
        let mut size = 0;
        for tex in self.textures.iter(){
            let s = tex.pixels.len();
            size += s;
        }
        let mut res = build_vec(size);
        let mut start = 0;
        for tex in self.textures.iter(){
            let len = tex.pixels.len();
            for j in 0..len{
                res[start + j] = tex.pixels[j];
            }
            start += len;
        }
        make_nonzero_len(&mut res);
        res
    }

    pub fn get_texture_params_buffer(&self) -> Vec<i32>{
        let mut res = build_vec(self.textures.len() * 3);
        let mut start = 0;
        for (i, tex) in self.textures.iter().enumerate(){
            res[i * 3 + 0] = start as i32;
            res[i * 3 + 1] = tex.width;
            res[i * 3 + 2] = tex.height;
            start += tex.pixels.len();
        }
        make_nonzero_len(&mut res);
        res
    }

    pub fn bufferize<T: SceneItem>(vec: &mut Vec<f32>, start: &mut usize, list: &Vec<T>, stride: usize){
        for (i, item) in list.iter().enumerate(){
            let off = i * stride;
            let data = item.get_data();
            for (j, float) in data.iter().enumerate(){
                vec[*start + off + j] = *float;
            }
        }
        *start += list.len() * stride;
    }

    pub fn add_texture(&mut self, name: String, path: String, ttype: TexType){
        if self.ghost_textures.get(&name).is_some(){
            println!("Error: Texture name is already used: {}", name);
            return;
        }
        self.ghost_textures.insert(name, (path,ttype));
    }

    fn actually_load_texture(&mut self, name: &str) -> bool{
        if self.textures_ids.get(name).is_some() { return true;}
        let (path,ttype);
        if let Some((pathv,ttypev)) = self.ghost_textures.get(name){
            path = pathv;
            ttype = ttypev;
        }else{
            println!("Error: Texture not found: {}.", name);
            return false;
        }
        let tex = if *ttype == TexType::Vector3c8bpc { TraceTex::vector_tex(path) }
        else { TraceTex::scalar_tex(path) };
        self.textures_ids.insert(name.to_string(), self.next_texture.inc());
        if let Ok(x) = tex {self.textures.push(x);}
        else { return false; }
        //c#: Info.Textures.Add((name, (uint)tex.Pixels.Length));
        true
    }

    pub fn get_texture(&mut self, name: String) -> i32{
        if !self.actually_load_texture(&name) { return 0; }
        if let Some(x) = self.textures_ids.get(&name){
            return x + 1;
        }
        0
    }

    pub fn set_skybox(&mut self, name: &str){
        if name == "" { 
            self.skybox = 0;
        }else if !self.actually_load_texture(name){
            self.skybox = 0;
        }else if let Some(x) = self.textures_ids.get(name){
            self.skybox = x + 1;
        }
    }

    pub fn add_light(&mut self, l: Light){ self.lights.push(l); }
    pub fn add_sphere(&mut self, s: Sphere){ self.spheres.push(s); }
    pub fn add_plane(&mut self, p: Plane){ self.planes.push(p); }
    pub fn add_box(&mut self, b: Box){ self.boxes.push(b); }
}
