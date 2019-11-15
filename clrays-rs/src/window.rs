use crate::state;

use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use std::time::Duration;

pub struct Window<T>{
    title: String,
    width: u32,
    height: u32,
    state: T
}

impl<T: state::State> Window<T>{
    pub fn new(title: &str, width: u32, height: u32) -> Self{
        Self { title: title.to_string(), width, height, state: T::new() }
    }

    pub fn run(&mut self, handle_input: fn(&mut sdl2::EventPump, &mut T)){
        let contex = sdl2::init().unwrap();
        let video_subsystem = contex.video().unwrap();

        let window = video_subsystem.window(&self.title, self.width, self.height)
            .position_centered()
            .build()
            .unwrap();

        let mut canvas = window.into_canvas().build().unwrap();

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        canvas.present();
        let mut event_pump = contex.event_pump().unwrap();
        loop {
            canvas.clear();
            handle_input(&mut event_pump, &mut self.state);
            if self.state.should_close() { break; }
            self.state.update(0.0);
            if self.state.should_close() { break; }
            canvas.present();
            std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
        }
    }
}

pub fn std_input_handler<T: state::State>(event_pump: &mut sdl2::EventPump, state: &mut T){
    for event in event_pump.poll_iter() {
        match event {
            Event::Quit {..} |
            Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                state.set_to_close();
                break;
            },
            _ => {}
        }
    }
}
