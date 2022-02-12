use crate::vec3::Vec3;
use crate::scene::{ Scene, SceneItem };

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

