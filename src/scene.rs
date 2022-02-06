use crate::vec3::{ Vec3, Orientation };
use crate::trace_tex::{ TexType, TraceTex };
use crate::misc::{ Incrementable, build_vec, make_nonzero_len };
use crate::info::Info;
use crate::bvh::Bvh;
use crate::mesh::Mesh;
use crate::aabb::AABB;
use crate::primitive::{ Primitive, Shape };
use crate::cpu::inter::{ Ray, RayHit, inter_plane, inter_sphere, inter_triangle };

use std::collections::HashMap;
use std::convert::TryInto;

pub trait SceneItem{
    fn get_data(&self) -> Vec<f32>;
    fn add(self, scene: &mut Scene);
}

pub trait Intersectable {
    fn intersect(&self, ray: Ray, hit: &mut RayHit);
}

pub type MaterialIndex = u32;

#[derive(Clone, PartialEq, Debug)]
pub struct Material{ // 62 bytes =  2*12 + 9*4 + 2
    pub col: Vec3,
    pub abs_fres: Vec3, // either absorption or fresnell colour depending on the situation
    pub reflectivity: f32,
    pub transparency: f32,
    pub refraction: f32,
    pub roughness: f32,
    pub texture: u32,
    pub normal_map: u32,
    pub roughness_map: u32,
    pub metalic_map: u32,
    pub tex_scale: f32,
    pub is_checkerboard: bool,
    pub is_dielectric: bool,
    pub emittance: f32,
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
            abs_fres: Vec3::BLACK,
            reflectivity: 0.0,
            transparency: 0.0,
            refraction: 1.0,
            roughness: 1.0,
            texture: 0,
            normal_map: 0,
            roughness_map: 0,
            metalic_map: 0,
            tex_scale: 1.0,
            is_checkerboard: false,
            is_dielectric: false,
            emittance: 0.0,
        }
    }

    pub fn as_light(mut self, col: Vec3, emittance: f32) -> Self{
        self.col = col;
        self.emittance = emittance;
        self
    }

    pub fn as_checkerboard(mut self) -> Self{
        self.is_checkerboard = true;
        self
    }

    pub fn as_dielectric(mut self) -> Self{
        self.is_dielectric = true;
        self
    }

    pub fn as_conductor(mut self) -> Self{
        self.is_dielectric = false;
        self
    }

    pub fn with_colour(mut self, col: Vec3) -> Self{
        self.col = col;
        self
    }

    pub fn with_reflectivity(mut self, refl: f32) -> Self{
        self.reflectivity = refl;
        self
    }

    pub fn with_transparency(mut self, tran: f32) -> Self{
        self.transparency = tran;
        self
    }

    pub fn with_refraction(mut self, refr: f32) -> Self{
        self.refraction = refr;
        self
    }

    pub fn with_absorption(mut self, absorption: Vec3) -> Self{
        let ab = Vec3{
            x: (1.0 - absorption.x).ln(),
            y: (1.0 - absorption.y).ln(),
            z: (1.0 - absorption.z).ln(),
        };
        self.abs_fres = ab;
        self
    }

    pub fn with_specular(mut self, spec: Vec3) -> Self{
        self.abs_fres = spec;
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

    pub fn add_to_scene(self, scene: &mut Scene) -> MaterialIndex{
        scene.get_mat_index(self)
    }

}

impl SceneItem for Material{
    fn get_data(&self) -> Vec<f32>{
        let refraction = if self.is_dielectric { self.refraction } else { -1.0 };
        vec![
            self.col.x, self.col.y, self.col.z, self.reflectivity,
            self.abs_fres.x, self.abs_fres.y, self.abs_fres.z, refraction,
            self.roughness, self.emittance,
            self.texture as f32, self.normal_map as f32,
            self.roughness_map as f32, self.metalic_map as f32,
            self.tex_scale
        ]
    }

    fn add(self, scene: &mut Scene){
        scene.get_mat_index(self);
    }
}

pub struct Plane{
    pub pos: Vec3,
    pub nor: Vec3,
    pub mat: MaterialIndex,
}

impl SceneItem for Plane{
    fn get_data(&self) -> Vec<f32>{
        vec![
            self.pos.x, self.pos.y, self.pos.z,
            self.nor.x, self.nor.y, self.nor.z,
            self.mat as f32
        ]
    }

    fn add(self, scene: &mut Scene){
        scene.add_plane(self);
    }
}

impl Intersectable for Plane {
    #[inline]
    fn intersect(&self, ray: Ray, hit: &mut RayHit) {
        inter_plane(ray, self, hit);
    }
}

pub struct Sphere{
    pub pos: Vec3,
    pub rad: f32,
    pub mat: MaterialIndex,
}

impl SceneItem for Sphere{
    fn get_data(&self) -> Vec<f32>{
        vec![ self.pos.x, self.pos.y, self.pos.z, self.rad, self.mat as f32 ]
    }

    fn add(self, scene: &mut Scene){
        scene.add_sphere(self);
    }
}

impl Intersectable for Sphere {
    #[inline]
    fn intersect(&self, ray: Ray, hit: &mut RayHit){
        inter_sphere(ray, self, hit);
    }
}

#[derive(Default,Debug,Clone)]
pub struct Triangle{ // 37 byte
    pub a: Vec3, // Vec3: 12 byte
    pub b: Vec3, // Vec3: 12 byte
    pub c: Vec3, // Vec3: 12 byte
    pub mat: MaterialIndex, // u32: 4 byte
}

impl SceneItem for Triangle{
    fn get_data(&self) -> Vec<f32>{
        vec![
            self.a.x, self.a.y, self.a.z,
            self.b.x, self.b.y, self.b.z,
            self.c.x, self.c.y, self.c.z,
            self.mat as f32
        ]
    }

    fn add(self, scene: &mut Scene){
        scene.add_triangle(self);
    }
}

impl Intersectable for Triangle {
    #[inline]
    fn intersect(&self, ray: Ray, hit: &mut RayHit){
        inter_triangle(ray, self, hit);
    }
}

pub type MeshIndex = u32;

#[derive(Clone, Copy)]
pub struct Model{
    pub pos: Vec3,
    // pub scale: Vec3,
    pub rot: Vec3,
    pub mat: MaterialIndex,
    pub mesh: MeshIndex
}

impl SceneItem for Model{
    fn add(self, scene: &mut Scene){
        scene.add_model(self);
    }

    fn get_data(&self) -> Vec<f32>{ vec![] }
}

pub struct Light{
    pub pos: Vec3,
    pub intensity: f32,
    pub col: Vec3,
}

impl SceneItem for Light{
    fn get_data(&self) -> Vec<f32>{
        vec![
            self.pos.x, self.pos.y, self.pos.z, self.intensity,
            self.col.x, self.col.y, self.col.z
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
    pub ori: Orientation,
    pub move_sensitivity: f32,
    pub look_sensitivity: f32,
    pub fov: f32,
    pub chromatic_aberration_shift: usize,
    pub chromatic_aberration_strength: f32,
    pub vignette_strength: f32,
    pub angle_radius: f32,
    pub distortion_coefficient: f32
}

#[derive(Clone, Copy, PartialEq)]
pub enum SceneType{
    Whitted,
    GI,
}

pub struct Scene{
    pub stype: SceneType,
    pub spheres: Vec<Sphere>,
    pub planes: Vec<Plane>,
    pub lights: Vec<Light>,
    pub mats: Vec<Material>,
    pub triangles: Vec<Triangle>,
    pub meshes: Vec<Mesh>,
    pub models: Vec<Model>,
    pub primitives: Vec<Primitive>,
    pub sub_bvhs: Vec<Bvh>,
    pub top_bvh: Bvh,
    scene_params: [u32; Self::SCENE_PARAM_SIZE],
    next_texture: u32,
    ghost_textures: HashMap<String, (String, TexType)>,
    textures_ids: HashMap<String, u32>,
    indexed_textures: Vec<(String, TexType, String)>,
    textures: Vec<TraceTex>,
    skybox: u32,
    pub sky_col: Vec3,
    pub sky_intensity: f32,
    pub sky_min: f32,
    pub sky_pow: f32,
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
    const SCENE_PARAM_SIZE: usize = 7 * 2 + Self::SCENE_SIZE as usize;
    const MATERIAL_SIZE: u32 = 15;
    const MATERIAL_INDEX_SIZE: u32 = 1;
    const LIGHT_SIZE: u32 = 7;
    const PLANE_SIZE: u32 = 6 + Self::MATERIAL_INDEX_SIZE;
    const SPHERE_SIZE: u32 = 4 + Self::MATERIAL_INDEX_SIZE;
    const TRIANGLE_SIZE: u32 = 9 + Self::MATERIAL_INDEX_SIZE;

    pub fn new() -> Self{
        Self{
            stype: SceneType::GI,
            spheres: Vec::new(),
            planes: Vec::new(),
            triangles: Vec::new(),
            meshes: Vec::new(),
            models: Vec::new(),
            primitives: Vec::new(),
            sub_bvhs: Vec::new(),
            top_bvh: Bvh::default(),
            lights: Vec::new(),
            mats: vec![Material::basic()],
            scene_params: [0; Self::SCENE_PARAM_SIZE],
            next_texture: 0,
            ghost_textures: HashMap::new(),
            textures_ids: HashMap::new(),
            indexed_textures: Vec::new(),
            textures: Vec::new(),
            skybox: 0,
            sky_col: Vec3::ONE,
            sky_intensity: 1.0,
            sky_min: 0.1,
            sky_pow: 1.0,
            sky_box: 0,
            cam: Camera{
                pos: Vec3::ZERO,
                dir: Vec3::BACKWARD,
                ori: Orientation { yaw: 0.0, roll: 0.0 },
                move_sensitivity: 0.1,
                look_sensitivity: 0.05,
                fov: 90.0,
                chromatic_aberration_shift: 2,
                chromatic_aberration_strength: 0.2,
                vignette_strength: 0.1,
                angle_radius: 0.0,
                distortion_coefficient: 1.0
            },
        }
    }

    pub fn get_scene_buffer(&mut self) -> Vec<f32>{
        let mut len = 0;
        len += self.mats.len() * Self::MATERIAL_SIZE as usize;
        len += self.lights.len() * Self::LIGHT_SIZE as usize;
        len += self.planes.len() * Self::PLANE_SIZE as usize;
        len += self.spheres.len() * Self::SPHERE_SIZE as usize;
        len += self.triangles.len() * Self::TRIANGLE_SIZE as usize;
        let mut res = build_vec(len);
        let mut i = 0;
        Self::bufferize(&mut res, &mut i, &self.mats, Self::MATERIAL_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.lights, Self::LIGHT_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.planes, Self::PLANE_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.spheres, Self::SPHERE_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.triangles, Self::TRIANGLE_SIZE as usize);
        make_nonzero_len(&mut res);
        res
    }

    pub fn get_scene_params_buffer(&mut self) -> Vec<u32>{
        let mut i = 0;
        self.scene_params[0] = self.mats.len() as u32;
        self.scene_params[1] = i; i += self.mats.len() as u32 * Self::MATERIAL_SIZE;

        self.scene_params[2] = self.lights.len() as u32;
        self.scene_params[3] = i; i += self.lights.len() as u32 * Self::LIGHT_SIZE;

        self.scene_params[4] = self.planes.len() as u32;
        self.scene_params[5] = i; i += self.planes.len() as u32 * Self::PLANE_SIZE;

        self.scene_params[6] = self.spheres.len() as u32;
        self.scene_params[7] = i; i += self.spheres.len() as u32 * Self::SPHERE_SIZE;

        self.scene_params[8] = self.triangles.len() as u32;
        self.scene_params[9] = i; //i += self.triangles.len() as u32 * Self::TRIANGLE_SIZE;

        //scene
        self.scene_params[10] = self.skybox;
        self.put_in_scene_params(11, self.sky_col);
        self.scene_params[14] = self.sky_intensity.to_bits() as u32;
        self.scene_params[15] = self.sky_min.to_bits() as u32;
        self.scene_params[16] = self.sky_pow.to_bits() as u32;
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

    pub fn get_bvh_buffer(&self) -> Vec<u32>{
        let bvhs = 1 + self.sub_bvhs.len();
        let mut buffer = vec![0; 2 + bvhs * 2];

        buffer[2] = buffer.len() as u32; // bvh start
        buffer[3] = 0; // top level doesn't have a mesh
        for v in &self.top_bvh.vertices{
            buffer.push(v.bound.min.x.to_bits() as u32);
            buffer.push(v.bound.min.y.to_bits() as u32);
            buffer.push(v.bound.min.z.to_bits() as u32);
            buffer.push(v.bound.max.x.to_bits() as u32);
            buffer.push(v.bound.max.y.to_bits() as u32);
            buffer.push(v.bound.max.z.to_bits() as u32);
            buffer.push(v.left_first.try_into().unwrap());
            buffer.push(v.count.try_into().unwrap());
        }

        for (i, sbvh) in self.sub_bvhs.iter().enumerate(){
            buffer[(i + 1) * 2 + 2] = buffer.len() as u32; // bvh start
            buffer[(i + 1) * 2 + 3] = self.meshes[sbvh.mesh_index as usize].start.try_into().unwrap(); // bvh mesh
            for v in &sbvh.vertices{
                buffer.push(v.bound.min.x.to_bits() as u32);
                buffer.push(v.bound.min.y.to_bits() as u32);
                buffer.push(v.bound.min.z.to_bits() as u32);
                buffer.push(v.bound.max.x.to_bits() as u32);
                buffer.push(v.bound.max.y.to_bits() as u32);
                buffer.push(v.bound.max.z.to_bits() as u32);
                buffer.push(v.left_first.try_into().unwrap());
                buffer.push(v.count.try_into().unwrap());
            }
        }

        buffer[0] = buffer.len() as u32; // start primivites
        for prim in &self.primitives{
            buffer.push(prim.shape_type as u32);
            buffer.push(prim.index.try_into().unwrap());
        }

        buffer[1] = buffer.len() as u32; // start models
        for model in &self.models{
            buffer.push(model.pos.x.to_bits() as u32);
            buffer.push(model.pos.y.to_bits() as u32);
            buffer.push(model.pos.z.to_bits() as u32);
            buffer.push(model.rot.x.to_bits() as u32);
            buffer.push(model.rot.y.to_bits() as u32);
            buffer.push(model.rot.z.to_bits() as u32);
            buffer.push(model.mat);
            buffer.push(model.mesh);
        }
        buffer
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

    pub fn get_mat_index(&mut self, mat: Material) -> MaterialIndex {
        if let Some(i) = self.mats.iter().position(|m| *m == mat) {
            i as MaterialIndex
        } else {
            self.mats.push(mat);
            assert!(self.mats.len() < MaterialIndex::MAX as usize);
            (self.mats.len() - 1) as MaterialIndex
        }
    }

    pub fn add_model(&mut self, model: Model) {
        self.models.push(model);
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

    pub fn set_sky_intensity(&mut self, int: f32, min: f32, pow: f32){
        self.sky_intensity = int;
        self.sky_min = min;
        self.sky_pow = pow;
    }

    pub fn add_light(&mut self, l: Light){
        if self.stype == SceneType::GI { return; }
        self.lights.push(l);
    }

    pub fn is_not_ok(&self, mat: MaterialIndex) -> bool{
        let e = self.mats[mat as usize].emittance;
        e > 0.0 && self.stype == SceneType::Whitted
    }

    pub fn add_sphere(&mut self, s: Sphere){
        if self.is_not_ok(s.mat) { return; }
        self.spheres.push(s);
    }

    pub fn add_plane(&mut self, p: Plane){
        if self.is_not_ok(p.mat) { return; }
        self.planes.push(p);
    }

    pub fn add_triangle(&mut self, t: Triangle){
        if self.is_not_ok(t.mat) { return; }
        self.triangles.push(t);
    }

    pub fn add_mesh(&mut self, mesh_name: String) -> MeshIndex {
        if let Some(i) = self.meshes.iter().position(|m| *m.name == mesh_name) {
            i as u32
        } else {
            assert!(self.meshes.len() < MeshIndex::MAX as usize);
            // todo: mesh references to index of first triangle, including count
            let mut triangles = Mesh::load_model(&*mesh_name);
            let mesh = Mesh {
                name: mesh_name,
                start: self.triangles.len(),
                count: triangles.len()
            };
            let bvh = Bvh::from_mesh(self.meshes.len() as MeshIndex, &mut triangles, 12);
            for tri in triangles {
                self.add_triangle(tri);
            }
            self.sub_bvhs.push(bvh);
            self.meshes.push(mesh);
            (self.meshes.len() - 1) as MeshIndex
        }
    }

    #[inline]
    pub fn get_mesh_triangle(&self, mesh: &Mesh, index: usize) -> &Triangle{
        &self.triangles[mesh.start + index]
    }

    #[inline]
    pub fn gen_top_bvh(&mut self) {
        // gather aabbs
        let mut prims: Vec<Primitive> = vec![];
        let mut aabbs: Vec<AABB> = vec![];
        // build primitives
        // spheres
        for (i, sphere) in self.spheres.iter().enumerate() {
            aabbs.push(AABB::from_point_radius(sphere.pos, sphere.rad));
            prims.push(Primitive{
                shape_type: Shape::SPHERE,
                index: i
            });
        }
        for (i, model) in self.models.iter().enumerate() {
            let sub_bvh: &Bvh = &self.sub_bvhs[model.mesh as usize];
            let aabb = sub_bvh.vertices.first().unwrap().bound;
            // rotate aabb and recompute surrounding aabb

            // obtain 8 corner points
            let a = aabb.min;
            let b = aabb.max;
            let d = b.subed(a);

            let mut points = vec![a; 8];
            points[1].x += d.x; points[2].x += d.x; points[5].x += d.x; points[6].x += d.x;
            points[4].y += d.y; points[5].y += d.y; points[6].y += d.y; points[7].y += d.y;
            points[2].z += d.z; points[3].z += d.z; points[6].z += d.z; points[7].z += d.z;

            points = points.iter_mut().map(|point| point
                .subed(aabb.midpoint()).yawed(model.rot.x).added(aabb.midpoint()) // apply model rotation
                .added(model.pos) // add model position
            ).collect();

            let aabb = AABB::from_points(&points);
            aabbs.push(aabb);
            prims.push(Primitive {
                shape_type: Shape::MODEL,
                index: i as usize
            });
        }
        // build bvh over aabbs
        self.top_bvh = Bvh::from_primitives(&mut aabbs, &mut prims);
        self.primitives = prims;
    }
}
