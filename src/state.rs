use crate::scene::Scene;
use crate::vec3::Vec3;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use std::f32::consts::{ PI, FRAC_PI_2 };

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoopRequest{
    Continue,
    Stop,
    Export,
}

const KEYS_AMOUNT: usize = 13;
pub type Keymap = [Keycode; KEYS_AMOUNT];

#[macro_export]
macro_rules! build_keymap{
    ($mfo:ident,$mba:ident,$mle:ident,$mri:ident,$mup:ident,$mdo:ident,
     $lup:ident,$ldo:ident,$lle:ident,$lri:ident,$foc:ident,$scr:ident,
     $bvh:ident) => {
        [Keycode::$mfo, Keycode::$mba, Keycode::$mle, Keycode::$mri, Keycode::$mup, Keycode::$mdo,
         Keycode::$lup, Keycode::$ldo, Keycode::$lle, Keycode::$lri, Keycode::$foc, Keycode::$scr,
         Keycode::$bvh]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMode{
    Full,
    Reduced,
    None,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Settings{
    pub aa_samples: usize,
    pub max_reduced_ms: f32,
    pub start_in_focus_mode: bool,
    pub max_render_depth: u8,
    pub calc_frame_energy: bool,
}

impl Default for Settings{
    fn default() -> Self{
        Self{
            aa_samples: 4,
            max_reduced_ms: 50.0,
            start_in_focus_mode: false,
            max_render_depth: 5,
            calc_frame_energy: false,
        }
    }
}

impl Settings{
    pub fn start_aa(&self) -> usize{
        if self.start_in_focus_mode{
            self.aa_samples
        } else {
            1
        }
    }
}

#[derive(Clone, Debug)]
pub struct State{
    pub key_map: Keymap,
    keys: [bool; KEYS_AMOUNT],
    pub render_mode: RenderMode,
    pub last_frame: RenderMode,
    pub reduced_rate: usize,
    pub aa: usize,
    pub samples_taken: usize,
    pub show_bvh: bool,
    pub settings: Settings,
    pub moved: bool,
    pub frame_energy: f32,
}

impl State{
    pub fn new(key_map: Keymap, settings: Settings) -> Self{
        Self{
            key_map,
            keys: [false; KEYS_AMOUNT],
            render_mode: RenderMode::Reduced,
            last_frame: RenderMode::None,
            reduced_rate: 4,
            aa: settings.start_aa(),
            samples_taken: 0,
            settings,
            show_bvh: false,
            moved: true,
            frame_energy: 0.0,
        }
    }

    pub fn toggle_focus_mode(&mut self){
        self.samples_taken = 0;
        self.render_mode = RenderMode::Reduced;
        self.aa = if self.aa == 1 {
            self.settings.aa_samples
        } else {
            1
        }
    }

    pub fn toggle_show_bvh(&mut self){
        self.show_bvh = !self.show_bvh;
        self.render_mode = RenderMode::Reduced;
        self.samples_taken = 0;
    }
}

pub type InputFn = fn (_events: &[Event], _scene: &mut Scene, _state: &mut State) -> LoopRequest;
pub type UpdateFn = fn (_dt: f32, _state: &mut State) -> LoopRequest;

pub fn std_update_fn(dt: f32, state: &mut State) -> LoopRequest {
    if state.last_frame == RenderMode::Reduced && dt > state.settings.max_reduced_ms{
        state.reduced_rate += 1;
    }
    LoopRequest::Continue
}

pub fn log_update_fn(dt: f32, state: &mut State) -> LoopRequest {
    if state.last_frame == RenderMode::Reduced{
        println!("{:?}({}): {} ms, ", state.last_frame, state.reduced_rate, dt);
    } else if state.last_frame == RenderMode::Full{
        println!("{:?}({}): {} ms, ", state.last_frame, state.samples_taken, dt);
        if state.frame_energy > 0.01{
            println!("Frame Energy: {}", state.frame_energy);
        }
    }
    std_update_fn(dt, state)
}

pub fn std_input_fn(events: &[Event], _: &mut Scene, _: &mut State) -> LoopRequest{
    for event in events.iter() {
        match event {
            Event::Quit {..} |
            Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                return LoopRequest::Stop;
            },
            _ => {}
        }
    }
    LoopRequest::Continue
}

pub fn fps_input_fn(events: &[Event], scene: &mut Scene, state: &mut State) -> LoopRequest{
    let cam = &mut scene.cam;
    let old_pos = cam.pos;
    let old_dir = cam.dir;

    for event in events.iter() {
        match event {
            Event::Quit {..} |
            Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                return LoopRequest::Stop;
            },
            Event::KeyDown { keycode: Some(x), .. } if *x == state.key_map[10] => {
                state.toggle_focus_mode();
            },
            Event::KeyDown { keycode: Some(x), .. } if *x == state.key_map[11] => {
                return LoopRequest::Export;
            },
            Event::KeyDown { keycode: Some(x), .. } if *x == state.key_map[12] => {
                print!("Toggle BVH rendering");
                state.toggle_show_bvh();
            },
            Event::KeyDown { keycode: Some(x), repeat: false, .. } => {
                for (i, binding) in state.key_map.iter().enumerate(){
                    if x == binding{
                        state.keys[i] = true;
                    }
                }
            },
            Event::KeyUp { keycode: Some(x), repeat: false, .. } => {
                for (i, binding) in state.key_map.iter().enumerate(){
                    if x == binding{
                        state.keys[i] = false;
                    }
                }
            },
            _ => {},
        }
    }
    let ms = cam.move_sensitivity;
    let ls = cam.look_sensitivity;
    for (i, active) in state.keys.iter().enumerate(){
        if !active { continue; }
        match i {
            0 => { // Move Forward; Move into camera direction
                cam.pos.add(cam.dir.scaled(ms));
            },
            1 => { // Move Backward; Move opposite camera direction
                cam.pos.add(cam.dir.neged().scaled(ms));
            },
            2 => { // Move Left; Move camera direction crossed z-axis, negated
                cam.pos.add(cam.dir.crossed(Vec3::UP).neged().scaled(ms));
            },
            3 => { // Move Right; Move camera direction crossed z-axis
                cam.pos.add(cam.dir.crossed(Vec3::UP).scaled(ms));
            },
            4 => { // Move Up; Move camera direction crossed x-axis
                cam.pos.add(Vec3::UP.neged().scaled(ms));
            },
            5 => { // Move Down; Move camera direction crossed x-axis
                cam.pos.add(Vec3::DOWN.neged().scaled(ms));
            },
            6 => { // Look Up;
                cam.ori.roll = (cam.ori.roll + ls).min(FRAC_PI_2).max(-FRAC_PI_2);
                cam.dir = Vec3::from_orientation(&cam.ori);
            },
            7 => { // Look Down;
                cam.ori.roll = (cam.ori.roll - ls).min(FRAC_PI_2).max(-FRAC_PI_2);
                cam.dir = Vec3::from_orientation(&cam.ori);
            },
            8 => { // Look Left;
                cam.ori.yaw -= ls;
                if cam.ori.yaw < -PI {
                    cam.ori.yaw += 2.0 * PI;
                }
                cam.dir = Vec3::from_orientation(&cam.ori);
            },
            9 => { // Look Right;
                cam.ori.yaw += ls;
                if cam.ori.yaw > PI {
                    cam.ori.yaw -= 2.0 * PI;
                }
                cam.dir = Vec3::from_orientation(&cam.ori);
            },
            _ => {},
        }
    }
    let moved = old_pos != cam.pos || old_dir != cam.dir;
    state.render_mode = match (moved, state.render_mode){
        (true, _) => RenderMode::Reduced,
        (false, RenderMode::Reduced) => RenderMode::Full,
        _ => RenderMode::None,
    };
    state.moved = moved;
    LoopRequest::Continue
}
