use crate::scene::RenderType;

use serde::Deserialize;

use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Deserialize, Clone)]
pub struct Config{
    pub title: Option<String>,
    pub gpu: bool,
    pub render_type: String,
}

#[derive(Clone)]
pub struct ParsedConf{
    pub title: Option<String>,
    pub gpu: bool,
    pub render_type: RenderType,
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
        Ok(ParsedConf{
            title, gpu, render_type,
        })
    }
}
