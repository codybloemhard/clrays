use crate::vec3::{ Vec3, Orientation };
use crate::trace_tex::{ TexType, TraceTex };
use crate::misc::{ Incrementable, build_vec, make_nonzero_len };
use crate::info::Info;
use crate::bvh::Bvh;
use crate::mesh::Mesh;

use std::collections::HashMap;
use crate::aabb::AABB;
use crate::primitive::{Primitive, Shape};
use crate::cpu::inter::{Ray, RayHit};
use crate::consts::{EPSILON, UV_SPHERE, UV_PLANE};

type MaterialIndex = u8;

#[derive(Clone, PartialEq, Debug)]
pub struct Material{ // 62 bytes =  2*12 + 9*4 + 2
    pub col: Vec3,
    pub absorption: Vec3,
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
    pub is_dielectric: bool
}

impl Material{
    pub fn get_tex_scale(&self) -> f32{
        self.tex_scale
    }

    pub fn set_tex_scale(&mut self, v: f32){
        self.tex_scale = 1.0 / v;
    }

    pub fn add_to_scene(self, scene: &mut Scene) -> MaterialIndex{
        scene.add_material(self)
    }

    pub fn basic() -> Self{
        Self{
            col: Vec3::ONE.unhardened(0.05),
            absorption: Vec3::BLACK,
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

    pub fn with_transparency(mut self, tran: f32) -> Self{
        self.transparency = tran;
        self
    }

    pub fn with_refraction(mut self, refr: f32) -> Self{
        self.refraction = refr;
        self
    }

    pub fn with_absorption(mut self, absorption: Vec3) -> Self{
        self.absorption = absorption;
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

    pub fn as_checkerboard(mut self) -> Self{
        self.is_checkerboard = true;
        self
    }

    pub fn as_dielectric(mut self) -> Self{
        self.is_dielectric = true;
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

pub trait Bufferizable{
    fn get_data(&self) -> Vec<f32>;
}

pub trait SceneItem{
    fn add(self, scene: &mut Scene);
    // fn bound(&self) -> AABB;
}

pub struct Plane{
    pub pos: Vec3,
    pub nor: Vec3,
    pub mat: MaterialIndex,
}

impl Bufferizable for Plane {
    fn get_data(&self) -> Vec<f32> {
        vec![
            self.pos.x, self.pos.y, self.pos.z,
            self.nor.x, self.nor.y, self.nor.z,
            self.mat as f32
        ]
    }
}

impl Intersectable for Plane {
    // ray-plane intersection
    #[inline]
    fn intersect(&self, ray: Ray, hit: &mut RayHit) {
        let divisor = Vec3::dot(ray.dir, self.nor);
        if divisor.abs() < EPSILON { return; }
        let planevec = Vec3::subed(self.pos, ray.pos);
        let t = Vec3::dot(planevec, self.nor) / divisor;
        if t < EPSILON { return; }
        if t > hit.t { return; }
        hit.t = t;
        hit.pos = ray.pos.added(ray.dir.scaled(t));
        hit.nor = self.nor;
        hit.mat = self.mat;
        hit.uvtype = UV_PLANE;
    }
}

impl SceneItem for Plane{
    fn add(self, scene: &mut Scene){
        scene.add_plane(self);
    }

    // fn bound(&self) -> AABB{
    //     panic!("Plane\'s cannot be bound!");
    // }
}

pub struct Sphere{
    pub pos: Vec3,
    pub rad: f32,
    pub mat: MaterialIndex,
}

impl Bufferizable for Sphere {
    fn get_data(&self) -> Vec<f32> {
        vec![self.pos.x, self.pos.y, self.pos.z, self.rad, self.mat as f32]
    }
}
impl SceneItem for Sphere{
    fn add(self, scene: &mut Scene){
        scene.add_sphere(self);
    }
}
impl Intersectable for Sphere {
    #[inline]
    fn intersect(&self, ray: Ray, hit: &mut RayHit){
        let l = Vec3::subed(self.pos, ray.pos);
        let tca = Vec3::dot(ray.dir, l);
        let d = tca*tca - Vec3::dot(l, l) + self.rad*self.rad;
        if d < 0.0 { return; }
        let dsqrt = d.sqrt();
        let mut t = tca - dsqrt;
        if t < 0.0 {
            t = tca + dsqrt;
            if t < 0.0 { return; }
        }
        if t > hit.t { return; }
        hit.t = t;
        hit.pos = ray.pos.added(ray.dir.scaled(t));
        hit.nor = Vec3::subed(hit.pos, self.pos).scaled(1.0 / self.rad);
        hit.mat = self.mat;
        hit.uvtype = UV_SPHERE;
    }
}


#[derive(Default,Debug,Clone)]
pub struct Triangle{ // 37 byte
    pub a: Vec3, // Vec3: 12 byte
    pub b: Vec3, // Vec3: 12 byte
    pub c: Vec3, // Vec3: 12 byte
}
impl Bufferizable for Triangle {
    fn get_data(&self) -> Vec<f32> {
        vec![
            self.a.x, self.a.y, self.a.z,
            self.b.x, self.b.y, self.b.z,
            self.c.x, self.c.y, self.c.z,
        ]
    }
}
impl Intersectable for Triangle {
    // ray-triangle intersection
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm?oldformat=true
    #[inline]
    #[allow(clippy::many_single_char_names)]
    fn intersect(&self, ray: Ray, hit: &mut RayHit){
        let edge1 = Vec3::subed(self.b, self.a);
        let edge2 = Vec3::subed(self.c, self.a);
        let h = Vec3::crossed(ray.dir, edge2);
        let a = Vec3::dot(edge1, h);
        if a > -EPSILON * 0.01 && a < EPSILON * 0.01 { return; } // ray parallel to tri
        let f = 1.0 / a;
        let s = Vec3::subed(ray.pos, self.a);
        let u = f * Vec3::dot(s, h);
        if !(0.0..=1.0).contains(&u) { return; }
        let q = Vec3::crossed(s, edge1);
        let v = f * Vec3::dot(ray.dir, q);
        if v < 0.0 || u + v > 1.0 { return; }
        let t = f * Vec3::dot(edge2, q);
        if t >= EPSILON && t < hit.t {
            hit.t = t;
            hit.pos = ray.pos.added(ray.dir.scaled(t));
            hit.nor = Vec3::crossed(edge1, edge2).normalized_fast();
        }
    }
}

pub trait Intersectable {
    fn intersect(&self, ray: Ray, hit: &mut RayHit);
}

pub type MeshIndex = u8;
pub type ModelIndex = u8;

impl Bufferizable for Mesh{
    fn get_data(&self) -> Vec<f32>{
        vec![self.start as f32,self.count as f32]
    }
}

#[derive(Clone, Copy)]
pub struct Model{
    pub pos: Vec3,
    // pub scale: Vec3,
    pub rot: Vec3,
    pub mat: MaterialIndex,
    pub mesh: MeshIndex
}
impl Bufferizable for Model{
    fn get_data(&self) -> Vec<f32>{
        // let t = vec![self.pos.get_data(),self.rot.get_data()].into_iter().flatten().collect();
        // t.push(self.mat as f32);
        // t.push(self.mesh as f32);
        // t
        vec![]
    }
}
impl SceneItem for Model{
    fn add(self, scene: &mut Scene){
        scene.add_model(self);
    }
}

pub struct Light{
    pub pos: Vec3,
    pub intensity: f32,
    pub col: Vec3,
}

impl Bufferizable for Light{
    fn get_data(&self) -> Vec<f32>{
        vec![
            self.pos.x, self.pos.y, self.pos.z,
            self.intensity, self.col.x, self.col.y, self.col.z
        ]
    }
}
impl SceneItem for Light{
    fn add(self, scene: &mut Scene){
        scene.add_light(self);
    }

    // fn bound(&self) -> AABB{
    //     panic!("Lights need not to be bound!");
    // }
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

pub struct Scene{
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
    const TRIANGLE_SIZE: u32 = 9 + Self::MATERIAL_SIZE;

    pub fn new() -> Self{
        Self{
            spheres: Vec::new(),
            planes: Vec::new(),
            mats: Vec::new(),
            triangles: Vec::new(),
            meshes: Vec::new(),
            models: Vec::new(),
            primitives: Vec::new(),
            sub_bvhs: Vec::new(),
            top_bvh: Bvh::default(),
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

    pub fn get_buffers(&mut self) -> Vec<f32>{
        let mut len = self.lights.len() * Self::LIGHT_SIZE as usize;
        len += self.spheres.len() * Self::SPHERE_SIZE as usize;
        len += self.planes.len() * Self::PLANE_SIZE as usize;
        len += self.triangles.len() * Self::TRIANGLE_SIZE as usize;
        let mut res = build_vec(len);
        let mut i = 0;
        Self::bufferize(&mut res, &mut i, &self.lights, Self::LIGHT_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.spheres, Self::SPHERE_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.planes, Self::PLANE_SIZE as usize);
        Self::bufferize(&mut res, &mut i, &self.triangles, Self::TRIANGLE_SIZE as usize);
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
        self.scene_params[9] = Self::TRIANGLE_SIZE;
        // self.scene_params[10] = self.triangles.len() as u32;
        // self.scene_params[11] = i; //i += self.triangles.len() as u32 * Self::TRIANGLE_SIZE;
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

    pub fn bufferize<T: Bufferizable>(vec: &mut Vec<f32>, start: &mut usize, list: &[T], stride: usize){
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

    pub fn add_material(&mut self, mat: Material) -> MaterialIndex {
        if let Some(i) = self.mats.iter().position(|m| *m == mat) {
            i as MaterialIndex
        } else {
            self.mats.push(mat);
            assert!(self.mats.len() < 255);
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

    pub fn add_light(&mut self, l: Light){ self.lights.push(l); }
    pub fn add_sphere(&mut self, s: Sphere){ self.spheres.push(s); }
    pub fn add_plane(&mut self, p: Plane){ self.planes.push(p); }
    pub fn add_triangle(&mut self, b: Triangle){ self.triangles.push(b); }

    pub fn add_mesh(&mut self, mesh_name: String) -> MeshIndex {
        if let Some(i) = self.meshes.iter().position(|m| *m.name == mesh_name) {
            i as u8
        } else {
            assert!(self.meshes.len() < 255);
            // todo: mesh references to index of first triangle, including count
            let mut triangles = Mesh::load_model(&*mesh_name);
            let mesh = Mesh {
                name: mesh_name,
                start: self.triangles.len(),
                count: triangles.len()
            };
            let bvh = Bvh::from_mesh(self.meshes.len() as u8, &mut triangles, 12);
            for tri in triangles {
                self.add_triangle(tri);
            }
            self.sub_bvhs.push(bvh);
            self.meshes.push(mesh);
            (self.meshes.len() - 1) as u8
        }
    }

    #[inline]
    pub fn get_mesh_triangle(&self, mesh: &Mesh, index: usize) -> &Triangle{
        &self.triangles[mesh.start + index]
    }

    #[inline]
    pub fn gen_top_bvh(&mut self) {
        let quality = 0;

        // gather aabbs
        let mut prims: Vec<Primitive> = vec![];
        let mut aabbs: Vec<AABB> = vec![];
        // build primitives
        // spheres
        for (i,sphere) in self.spheres.iter().enumerate() {
            aabbs.push(AABB::from_point_radius(sphere.pos, sphere.rad));
            prims.push(Primitive{
                shape_type: Shape::SPHERE,
                index: i
            });
        }
        for (i,model) in self.models.iter().enumerate() {
            let sub_bvh: &Bvh = &self.sub_bvhs[model.mesh as usize];
            let aabb = sub_bvh.vertices.first().unwrap().bound;
            // rotate aabb and recompute surrounding aabb

            // obtain 8 corner points
            let a = aabb.min;
            let b = aabb.max;
            let d = b.subed(a);

            let mut points = vec![a; 8];

            points[1].x += d.x;

            points[2].x += d.x;
            points[2].z += d.z;

            points[3].z += d.z;

            points[4].y += d.y;
            points[5].y += d.y;
            points[6].y += d.y;
            points[7].y += d.y;

            points[5].x += d.x;

            points[6].x += d.x;
            points[6].z += d.z;

            points[7].z += d.z;

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



    // pub fn either_sphere_or_triangle(&self, index: usize) -> Either<&Sphere, &Triangle>{
    //     let sl = self.spheres.len();
    //     if index < sl{ // is sphere
    //         Either::Left(&self.spheres[index])
    //     } else { // is triangle
    //         // figure out which model
    //         Either::Right(&self.triangles[index - sl])
    //     }
    // }
}

pub enum Either<T, S> { Left(T), Right(S) }
