use crate::scene::RenderType;

use serde::Deserialize;

use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Deserialize, Clone)]
pub struct Config{
    title: Option<String>,
    gpu: bool,
    render_type: String,
    width: u32,
    height: u32,
    aa_samples: Option<usize>,
    render_depth: Option<u8>,
    max_reduced_ms: Option<f32>,
    start_in_focus_mode: Option<bool>,
    frame_energy: Option<bool>,
}

#[derive(Clone)]
pub struct ParsedConf{
    pub title: Option<String>,
    pub gpu: bool,
    pub render_type: RenderType,
    pub w: u32,
    pub h: u32,
    pub aa_samples: usize,
    pub render_depth: u8,
    pub max_reduced_ms: f32,
    pub start_in_focus_mode: bool,
    pub frame_energy: bool,
}

impl Config{
    pub fn read(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn parse(self) -> Result<ParsedConf, String>{
        let title = self.title;
        let gpu = self.gpu;
        let render_type = match self.render_type.to_lowercase().as_ref(){
            "gi" => RenderType::GI,
            "whitted" => RenderType::Whitted,
            _ => return Err(format!("Target '{}' is not supported!", self.render_type))
        };
        let w = if self.width == 0 { 1024 } else { self.width };
        let h = if self.height == 0 { 1024 } else { self.height };
        let aa_samples = self.aa_samples.unwrap_or(1).max(1);
        let render_depth = self.render_depth.unwrap_or(5).max(1);
        let max_reduced_ms = self.max_reduced_ms.unwrap_or(40.0).max(1.0);
        let start_in_focus_mode = self.start_in_focus_mode.unwrap_or(false);
        let frame_energy = self.frame_energy.unwrap_or(false);
        Ok(ParsedConf{
            title, gpu, render_type, w, h,
            aa_samples, render_depth, max_reduced_ms, start_in_focus_mode, frame_energy
        })
    }
}
