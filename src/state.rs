use crate::scene::Scene;
use crate::vec3::Vec3;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use std::f32::consts::{ PI, FRAC_PI_2 };

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoopRequest{
    Continue,
    Stop,
}

pub type Keymap = [Keycode; 10];

#[macro_export]
macro_rules! build_keymap{
    ($mfo:ident,$mba:ident,$mle:ident,$mri:ident,$mup:ident,$mdo:ident,
     $lup:ident,$ldo:ident,$lle:ident,$lri:ident) => {
        [Keycode::$mfo, Keycode::$mba, Keycode::$mle, Keycode::$mri, Keycode::$mup, Keycode::$mdo,
         Keycode::$lup, Keycode::$ldo, Keycode::$lle, Keycode::$lri]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMode{
    Full,
    Reduced,
    None,
}

#[derive(Clone, Debug)]
pub struct State{
    pub key_map: Keymap,
    keys: [bool; 10],
    pub render_mode: RenderMode,
}

impl State{
    pub fn new(key_map: Keymap) -> Self{
        Self{
            key_map,
            keys: [false; 10],
            render_mode: RenderMode::Reduced,
        }
    }
}

pub type InputFn = fn (_events: &[Event], _scene: &mut Scene, _state: &mut State) -> LoopRequest;
pub type UpdateFn = fn (_dt: f64) -> LoopRequest;

pub fn std_update_fn(_: f64) -> LoopRequest { LoopRequest::Continue }

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
    fn yaw_roll(yaw: f32, roll: f32) -> Vec3 {
        let a = roll;  // Up/Down
        let b = yaw;   // Left/Right
        Vec3 { x: a.cos() * b.sin(), y: a.sin(), z: -a.cos() * b.cos() }
    }

    let cam = &mut scene.cam;
    let old_pos = cam.pos;
    let old_dir = cam.dir;
    let keys = &mut state.keys;

    for event in events.iter() {
        match event {
            Event::Quit {..} |
            Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                return LoopRequest::Stop;
            },
            Event::KeyDown { keycode: Some(x), repeat: false, .. } => {
                for (i, binding) in state.key_map.iter().enumerate(){
                    if x == binding{
                        keys[i] = true;
                    }
                }
            },
            Event::KeyUp { keycode: Some(x), repeat: false, .. } => {
                for (i, binding) in state.key_map.iter().enumerate(){
                    if x == binding{
                        keys[i] = false;
                    }
                }
            },
            _ => {},
        }
    }
    let ms = cam.move_sensitivity;
    let ls = cam.look_sensitivity;
    for (i, active) in keys.iter().enumerate(){
        if !active { continue; }
        match i {
            0 => { // Move Forward; Move into camera direction
                cam.pos.add(cam.dir.scaled(ms));
                break;
            },
            1 => { // Move Backward; Move opposite camera direction
                cam.pos.add(cam.dir.neged().scaled(ms));
                break;
            },
            2 => { // Move Left; Move camera direction crossed z-axis, negated
                cam.pos.add(cam.dir.crossed(Vec3::UP).neged().scaled(ms));
                break;
            },
            3 => { // Move Right; Move camera direction crossed z-axis
                cam.pos.add(cam.dir.crossed(Vec3::UP).scaled(ms));
                break;
            },
            4 => { // Move Up; Move camera direction crossed x-axis
                cam.pos.add(cam.dir.crossed(Vec3::RIGHT).scaled(ms));
                break;
            },
            5 => { // Move Down; Move camera direction crossed x-axis
                cam.pos.add(cam.dir.crossed(Vec3::RIGHT).neged().scaled(ms));
                break;
            },
            6 => { // Look Up;
                cam.ori[1] = (cam.ori[1] + ls).min(FRAC_PI_2).max(-FRAC_PI_2);
                let yaw = cam.ori[0]; // Up/Down
                let roll = cam.ori[1]; // Left/Right
                cam.dir = yaw_roll(yaw, roll);
            },
            7 => { // Look Down;
                cam.ori[1] = (cam.ori[1] - ls).min(FRAC_PI_2).max(-FRAC_PI_2);
                let yaw = cam.ori[0]; // Up/Down
                let roll = cam.ori[1]; // Left/Right
                cam.dir = yaw_roll(yaw, roll);
            },
            8 => { // Look Left;
                cam.ori[0] -= ls;
                if cam.ori[0] < -PI {
                    cam.ori[0] += 2.0 * PI;
                }
                let yaw = cam.ori[0]; // Up/Down
                let roll = cam.ori[1]; // Left/Right
                cam.dir = yaw_roll(yaw, roll);
            },
            9 => { // Look Right;
                cam.ori[0] += ls;
                if cam.ori[0] > PI {
                    cam.ori[0] -= 2.0 * PI;
                }
                let yaw = cam.ori[0]; // Up/Down
                let roll = cam.ori[1]; // Left/Right
                cam.dir = yaw_roll(yaw, roll);
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
    LoopRequest::Continue
}
