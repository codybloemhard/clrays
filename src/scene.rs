use crate::vec3::Vec3;
use crate::trace_tex::{ TexType, TraceTex };
use crate::misc::{ Incrementable, build_vec, make_nonzero_len };
use crate::info::Info;

use std::collections::HashMap;

pub struct Material{
    pub col: Vec3,
    pub reflectivity: f32,
    pub roughness: f32,
    pub texture: u32,
    pub normal_map: u32,
    pub roughness_map: u32,
    pub metalic_map: u32,
    pub tex_scale: f32,
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
            col: Vec3::ONE.unhardened(0.05),
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

    pub fn with_texture(mut self, tex: u32) -> Self{
        self.texture = tex;
        self
    }

    pub fn with_normal_map(mut self, norm: u32) -> Self{
        self.normal_map = norm;
        self
    }

    pub fn with_roughness_map(mut self, roughm: u32) -> Self{
        self.roughness_map = roughm;
        self
    }

    pub fn with_metalic_map(mut self, metalm: u32) -> Self{
        self.metalic_map = metalm;
        self
    }

    pub fn with_tex_scale(mut self, v: f32) -> Self{
        self.tex_scale = 1.0 / v;
        self
    }
}

pub trait SceneItem{
    fn get_data(&self) -> Vec<f32>;
    fn add(self, scene: &mut Scene);
}

pub struct Plane{
    pub pos: Vec3,
    pub nor: Vec3,
    pub mat: Material,
}

impl SceneItem for Plane{
    fn get_data(&self) -> Vec<f32>{
        vec![
            self.pos.x, self.pos.y, self.pos.z,
            self.nor.x, self.nor.y, self.nor.z,
            self.mat.col.x, self.mat.col.y, self.mat.col.z,
            self.mat.reflectivity, self.mat.roughness,
            self.mat.texture as f32, self.mat.normal_map as f32,
            self.mat.roughness_map as f32, self.mat.metalic_map as f32,
            self.mat.tex_scale
        ]
    }

    fn add(self, scene: &mut Scene){
        scene.add_plane(self);
    }
}

pub struct Sphere{
    pub pos: Vec3,
    pub rad: f32,
    pub mat: Material,
}

impl SceneItem for Sphere{
    fn get_data(&self) -> Vec<f32>{
        vec![
            self.pos.x, self.pos.y, self.pos.z, self.rad,
            self.mat.col.x, self.mat.col.y, self.mat.col.z,
            self.mat.reflectivity, self.mat.roughness,
            self.mat.texture as f32, self.mat.normal_map as f32,
            self.mat.roughness_map as f32, self.mat.metalic_map as f32,
            self.mat.tex_scale
        ]
    }

    fn add(self, scene: &mut Scene){
        scene.add_sphere(self);
    }
}

pub struct Box{
    pub pos: Vec3,
    pub size: Vec3,
    pub mat: Material,
}

impl SceneItem for Box{
    fn get_data(&self) -> Vec<f32>{
        let hs = self.size.scaled(0.5);
        vec![
            self.pos.x - hs.x, self.pos.y - hs.y, self.pos.z - hs.z,
            self.pos.x + hs.x, self.pos.y + hs.y, self.pos.z + hs.z,
            self.mat.texture as f32, self.mat.normal_map as f32,
            self.mat.roughness_map as f32, self.mat.metalic_map as f32,
            self.mat.tex_scale
        ]
    }

    fn add(self, scene: &mut Scene){
        scene.add_box(self);
    }
}

pub struct Light{
    pub pos: Vec3,
    pub intensity: f32,
    pub col: Vec3,
}

impl SceneItem for Light{
    fn get_data(&self) -> Vec<f32>{
        vec![
            self.pos.x, self.pos.y, self.pos.z,
            self.intensity, self.col.x, self.col.y, self.col.z
        ]
    }

    fn add(self, scene: &mut Scene){
        scene.add_light(self);
    }
}

#[derive(Clone, Debug)]
pub struct Camera{
    pub pos: Vec3,
    pub dir: Vec3,
    pub ori: Vec<f32>,
    pub move_sensitivity: f32,
    pub look_sensitivity: f32,
    pub fov: f32,
    pub chromatic_aberration_shift: usize,
    pub chromatic_aberration_strength: f32,
    pub vignette_strength: f32,
}

pub struct Scene{
    pub spheres: Vec<Sphere>,
    pub planes: Vec<Plane>,
    pub boxes: Vec<Box>,
    pub lights: Vec<Light>,
    scene_params: [u32; Self::SCENE_PARAM_SIZE],
    next_texture: u32,
    ghost_textures: HashMap<String, (String, TexType)>,
    textures_ids: HashMap<String, u32>,
    indexed_textures: Vec<(String, TexType, String)>,
    textures: Vec<TraceTex>,
    skybox: u32,
    pub sky_col: Vec3,
    pub sky_intensity: f32,
    pub sky_box: u32,
    pub cam: Camera,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene{
    const SCENE_SIZE: u32 = 11;
    const SCENE_PARAM_SIZE: usize = 4 * 3 + Self::SCENE_SIZE as usize;
    const MATERIAL_SIZE: u32 = 10;
    const LIGHT_SIZE: u32 = 7;
    const SPHERE_SIZE: u32 = 4 + Self::MATERIAL_SIZE;
    const PLANE_SIZE: u32 = 6 + Self::MATERIAL_SIZE;
    const BOX_SIZE: u32 = 6 + Self::MATERIAL_SIZE;

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
            indexed_textures: Vec::new(),
            textures: Vec::new(),
            skybox: 0,
            sky_col: Vec3::ONE,
            sky_intensity: 0.0,
            sky_box: 0,
            cam: Camera{
                pos: Vec3::ZERO,
                dir: Vec3::BACKWARD,
                ori: vec![0.0,0.0],
                move_sensitivity: 0.1,
                look_sensitivity: 0.05,
                fov: 90.0,
                chromatic_aberration_shift: 2,
                chromatic_aberration_strength: 0.2,
                vignette_strength: 0.1,
            }
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

    pub fn get_params_buffer(&mut self) -> Vec<u32>{
        let mut i = 0;
        self.scene_params[0] = Self::LIGHT_SIZE;
        self.scene_params[1] = self.lights.len() as u32;
        self.scene_params[2] = i; i += self.lights.len() as u32 * Self::LIGHT_SIZE;
        self.scene_params[3] = Self::SPHERE_SIZE;
        self.scene_params[4] = self.spheres.len() as u32;
        self.scene_params[5] = i; i += self.spheres.len() as u32 * Self::SPHERE_SIZE;
        self.scene_params[6] = Self::PLANE_SIZE;
        self.scene_params[7] = self.planes.len() as u32;
        self.scene_params[8] = i; i += self.planes.len() as u32 * Self::PLANE_SIZE;
        self.scene_params[9] = Self::BOX_SIZE;
        self.scene_params[10] = self.boxes.len() as u32;
        self.scene_params[11] = i; //i += self.boxes.len() as u32 * Self::BOX_SIZE;
        //scene
        self.scene_params[12] = self.skybox;
        self.put_in_scene_params(13, self.sky_col);
        self.scene_params[16] = self.sky_intensity.to_bits() as u32;
        self.put_in_scene_params(17, self.cam.pos);
        self.put_in_scene_params(20, self.cam.dir);
        self.scene_params.to_vec()
    }

    fn put_in_scene_params(&mut self, i: usize, v: Vec3){
        self.scene_params[i    ] = v.x.to_bits() as u32;
        self.scene_params[i + 1] = v.y.to_bits() as u32;
        self.scene_params[i + 2] = v.z.to_bits() as u32;
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
            res[start..(len + start)].clone_from_slice(&tex.pixels[..len]);
            start += len;
        }
        make_nonzero_len(&mut res);
        res
    }

    pub fn get_texture_params_buffer(&self) -> Vec<u32>{
        let mut res = build_vec(self.textures.len() * 3);
        let mut start = 0;
        for (i, tex) in self.textures.iter().enumerate(){
            res[i * 3    ] = start as u32;
            res[i * 3 + 1] = tex.width as u32;
            res[i * 3 + 2] = tex.height as u32;
            start += tex.pixels.len();
        }
        make_nonzero_len(&mut res);
        res
    }

    pub fn bufferize<T: SceneItem>(vec: &mut Vec<f32>, start: &mut usize, list: &[T], stride: usize){
        for (i, item) in list.iter().enumerate(){
            let off = i * stride;
            let data = item.get_data();
            for (j, float) in data.iter().enumerate(){
                vec[*start + off + j] = *float;
            }
        }
        *start += list.len() * stride;
    }

    pub fn add_texture(&mut self, name: &str, path: &str, ttype: TexType){
        if self.ghost_textures.get(name).is_some(){
            println!("Error: Texture name is already used: {}", name);
            return;
        }
        self.ghost_textures.insert(name.to_string(), (path.to_string(), ttype));
    }

    pub fn get_texture(&mut self, name: &str) -> u32{
        if let Some(x) = self.textures_ids.get(name){
            x + 1
        } else if let Some((path, ttype)) = self.ghost_textures.get(name){
            let id = self.next_texture.inc_post();
            self.textures_ids.insert(name.to_string(), id);
            self.indexed_textures.push((path.clone(), *ttype, name.to_string()));
            id + 1
        } else {
            println!("Warning: could not look up texture \"{}\"", name);
            0
        }
    }

    pub fn pack_textures(&mut self, info: &mut Info){
        for (path, ttype, name) in std::mem::take(&mut self.indexed_textures){
            let tex = if ttype == TexType::Vector3c8bpc { TraceTex::vector_tex(&path) }
            else { TraceTex::scalar_tex(&path) };
            match tex{
                Ok(x) => {
                    info.textures.push((name, x.pixels.len() as u64));
                    self.textures.push(x);
                },
                Err(e) => {
                    println!("Error: could not create texture \"{}\": {:?}", name, e);
                }
            }
        }
        info.set_time_point("Loading textures");
    }

    pub fn set_skybox(&mut self, name: &str){
        self.skybox = self.get_texture(name);
        self.sky_box = self.skybox;
    }

    pub fn add_light(&mut self, l: Light){ self.lights.push(l); }
    pub fn add_sphere(&mut self, s: Sphere){ self.spheres.push(s); }
    pub fn add_plane(&mut self, p: Plane){ self.planes.push(p); }
    pub fn add_box(&mut self, b: Box){ self.boxes.push(b); }
}
