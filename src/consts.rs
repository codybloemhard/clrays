use crate::vec3::Vec3;

pub const GAMMA: f32 = 2.2;
pub const PI: f32 = std::f32::consts::PI;
pub const FRAC_2_PI: f32 = 0.5 * std::f32::consts::PI;
pub const FRAC_4_PI: f32 = 0.25 * std::f32::consts::PI;
pub const MAX_RENDER_DIST: f32 = 1000000.0;
pub const EPSILON: f32 = 0.001;
pub const AMBIENT: f32 = 0.05;

pub const WATER_ABSORPTION : Vec3 = Vec3 { x: 0.49, y: 0.1, z: 0.04 };
pub const AIR_ABSORPTION : Vec3 = Vec3 { x: 0.01, y: 0.01, z: 0.01 };

// https://nature.berkeley.edu/classes/eps2/wisc/ri.html
pub const AIR_REFRACTION: f32 = 1.0;
pub const WATER_REFRACTION: f32 = 1.33;
pub const PLASTIC_REFRACTION: f32 = 1.58; // 1.46 - 1.7
pub const GLASS_REFRACTION: f32 = 1.67; // 1.44 - 1.9
pub const DIAMOND_REFRACTION: f32 = 2.418;

pub const UV_PLANE: u8 = 0;
pub const UV_SPHERE: u8 = 1;

