use crate::scene::Scene;
use crate::vec3::Vec3;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use std::f32::consts::{ PI, FRAC_PI_2 };

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LoopRequest{
    Continue,
    Stop,
}

pub type InputFn = fn (_events: &[Event], _scene: &mut Scene) -> LoopRequest;
pub type UpdateFn = fn (_dt: f64) -> LoopRequest;

pub fn std_update_fn(_dt: f64) -> LoopRequest { LoopRequest::Continue }

pub fn std_input_fn(events: &[Event], _scene: &mut Scene) -> LoopRequest{
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

pub fn fps_input_fn(events: &[Event], scene: &mut Scene) -> LoopRequest{
    fn yaw_roll(yaw: f32, roll: f32) -> Vec3 {
        let a = roll;  // Up/Down
        let b = yaw;   // Left/Right
        Vec3 { x: a.cos() * b.sin(), y: a.sin(), z: -a.cos() * b.cos() }
    }

    let cam = &mut scene.cam;
    for event in events.iter() {
        match event {
            Event::Quit {..} |
            Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                return LoopRequest::Stop;
            },
            Event::KeyDown { keycode: Some(Keycode::W), .. } => {
                // Move Forward; Move into camera direction
                let s = cam.move_sensitivity;
                cam.pos.add(cam.dir.scaled(s));
                break;
            },
            Event::KeyDown { keycode: Some(Keycode::S), .. } => {
                // Move Backward; Move opposite camera direction
                let s = cam.move_sensitivity;
                cam.pos.add(cam.dir.neged().scaled(s));
                break;
            },
            Event::KeyDown { keycode: Some(Keycode::D), .. } => {
                // Move Right; Move camera direction crossed z-axis
                let s = cam.move_sensitivity;
                cam.pos.add(cam.dir.crossed(Vec3::UP).scaled(s));
                break;
            },
            Event::KeyDown { keycode: Some(Keycode::A), .. } => {
                // Move Left; Move camera direction crossed z-axis, negated
                let s = cam.move_sensitivity;
                cam.pos.add(cam.dir.crossed(Vec3::UP).neged().scaled(s));
                break;
            },
            Event::KeyDown { keycode: Some(Keycode::I), .. } => {
                // Look Up;
                let s = cam.look_sensitivity;
                cam.ori[1] = (cam.ori[1] + s).min(FRAC_PI_2).max(-FRAC_PI_2);
                let yaw = cam.ori[0]; // Up/Down
                let roll = cam.ori[1]; // Left/Right
                cam.dir = yaw_roll(yaw, roll);
            },
            Event::KeyDown { keycode: Some(Keycode::K), .. } => {
                // Look Down;
                let s = cam.look_sensitivity;
                cam.ori[1] = (cam.ori[1] - s).min(FRAC_PI_2).max(-FRAC_PI_2);
                let yaw = cam.ori[0]; // Up/Down
                let roll = cam.ori[1]; // Left/Right
                cam.dir = yaw_roll(yaw, roll);
            },
            Event::KeyDown { keycode: Some(Keycode::L), .. } => {
                // Look Right;
                let s = cam.look_sensitivity;
                cam.ori[0] += s;
                if cam.ori[0] > PI {
                    cam.ori[0] -= 2.0 * PI;
                }
                let yaw = cam.ori[0]; // Up/Down
                let roll = cam.ori[1]; // Left/Right
                cam.dir = yaw_roll(yaw, roll);
            },
            Event::KeyDown { keycode: Some(Keycode::J), .. } => {
                // Look Left;
                let s = cam.look_sensitivity;
                cam.ori[0] -= s;
                if cam.ori[0] < -PI {
                    cam.ori[0] += 2.0 * PI;
                }
                let yaw = cam.ori[0]; // Up/Down
                let roll = cam.ori[1]; // Left/Right
                cam.dir = yaw_roll(yaw, roll);
            },
            _ => {}
        }
    }
    LoopRequest::Continue
}
