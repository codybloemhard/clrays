use crate::scene::RenderType;

use serde::Deserialize;

use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Deserialize, Clone)]
pub struct Config{
    base: Base,
    cpu: Option<Cpu>,
}

pub struct ConfigParsed{
    pub base: BaseParsed,
    pub cpu: CpuParsed,
}

impl Config{
    pub fn read(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    pub fn parse(self) -> Result<ConfigParsed, String>{
        let base = self.base.parse()?;
        let cpu = self.cpu.unwrap_or_default().parse();
        Ok(ConfigParsed{
            base, cpu
        })
    }
}

#[derive(Deserialize, Clone)]
pub struct Base{
    title: Option<String>,
    gpu: bool,
    render_type: String,
    width: u32,
    height: u32,
    frame_energy: Option<bool>,
}

pub struct BaseParsed{
    pub title: Option<String>,
    pub gpu: bool,
    pub render_type: Option<RenderType>, // None means run tests and exit
    pub w: u32,
    pub h: u32,
    pub frame_energy: bool,
}

impl Base{
    pub fn parse(self) -> Result<BaseParsed, String>{
        let title = self.title;
        let gpu = self.gpu;
        let render_type = match self.render_type.to_lowercase().as_ref(){
            "gi" => Some(RenderType::GI),
            "whitted" => Some(RenderType::Whitted),
            "test" => None,
            _ => return Err(format!("Target '{}' is not supported!", self.render_type))
        };
        let w = if self.width == 0 { 1024 } else { self.width };
        let h = if self.height == 0 { 1024 } else { self.height };
        let frame_energy = self.frame_energy.unwrap_or(false);
        Ok(BaseParsed{
            title, gpu, render_type, w, h, frame_energy
        })
    }
}

#[derive(Deserialize, Clone)]
pub struct Cpu{
    aa_samples: Option<usize>,
    render_depth: Option<u8>,
    max_reduced_ms: Option<f32>,
    start_in_focus_mode: Option<bool>,
}

pub struct CpuParsed{
    pub aa_samples: usize,
    pub render_depth: u8,
    pub max_reduced_ms: f32,
    pub start_in_focus_mode: bool,
}

impl Default for Cpu{
    fn default() -> Self{
        Self{
            aa_samples: Some(8),
            render_depth: Some(5),
            max_reduced_ms: Some(40.0),
            start_in_focus_mode: Some(false),
        }
    }
}

impl Cpu{
    pub fn parse(self) -> CpuParsed{
        let aa_samples = self.aa_samples.unwrap_or(1).max(1);
        let render_depth = self.render_depth.unwrap_or(5).max(1);
        let max_reduced_ms = self.max_reduced_ms.unwrap_or(40.0).max(1.0);
        let start_in_focus_mode = self.start_in_focus_mode.unwrap_or(false);
        CpuParsed{
            aa_samples, render_depth, max_reduced_ms, start_in_focus_mode
        }
    }
}
