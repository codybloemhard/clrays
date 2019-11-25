use crate::state;
use crate::trace_processor::TraceProcessor;
use crate::misc::Incrementable;

use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use stopwatch::Stopwatch;

pub struct Window<T>{
    title: String,
    width: u32,
    height: u32,
    state: T,
    tracer: TraceProcessor,
}

impl<T: state::State> Window<T>{
    pub fn new(title: &str, width: u32, height: u32, tracer: TraceProcessor) -> Self{
        Self { title: title.to_string(), width, height, state: T::new(), tracer, }
    }

    pub fn run(&mut self, handle_input: fn(&mut sdl2::EventPump, &mut T)) -> Option<String>{
        let contex = match sdl2::init(){
            Result::Ok(x) => x,
            Result::Err(e) => return Some(e),
        };
        let video_subsystem = match contex.video(){
            Result::Ok(x) => x,
            Result::Err(e) => return Some(e),
        };
        let window = match video_subsystem.window(&self.title, self.width, self.height)
            .position_centered()
            .build(){
                Result::Ok(x) => x,
                Result::Err(e) => return Some(window_build_error_to_string(e)),
            };

        let mut canvas = match window.into_canvas().build(){
            Result::Ok(x) => x,
            Result::Err(e) => return Some(integer_ord_sdl_error_to_string(e)),
        };

        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        canvas.present();
        let mut event_pump = match contex.event_pump(){
            Result::Ok(x) => x,
            Result::Err(e) => return Some(e),
        };
        let mut frame = 0;
        let mut watch = Stopwatch::start_new();
        let mut elapsed = 0;
        loop {
            //canvas.clear();
            handle_input(&mut event_pump, &mut self.state);
            if self.state.should_close() { break; }
            self.state.update(0.0);
            if self.state.should_close() { break; }
            self.tracer.update();
            let int_tex = self.tracer.render().unwrap();
            if frame % 10 == 0{
                buffer_to_screen(int_tex, &mut canvas, self.width as usize);
                canvas.present();
            }
            let e = watch.elapsed_ms();
            println!("Frame: {} in {} ms.", frame.inc_post(), e - elapsed);
            elapsed = e;
            //std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
        }
        None
    }
}

pub fn buffer_to_screen(int_tex: &[i32], canvas: &mut sdl2::render::Canvas<sdl2::video::Window>, w: usize){
    for (i,int) in int_tex.iter().enumerate(){
        let y = i / w;
        let x = i - (y * w);
        let r = (int >> 16) & 0xff;
        let g = (int >> 8) & 0xff;
        let b = (int >> 0) & 0xff;
        let col = Color::RGB(r as u8, g as u8, b as u8);
        canvas.set_draw_color(col);
        canvas.draw_point(sdl2::rect::Point::new(x as i32, y as i32)).expect("Error: sdl2 could not draw point.");
    }
}

pub fn window_build_error_to_string(wbe: sdl2::video::WindowBuildError) -> String{
    match wbe{
        sdl2::video::WindowBuildError::WidthOverflows(x) => format!("sdl2: WindowBuildError: WidthOverflows: {}", x),
        sdl2::video::WindowBuildError::HeightOverflows(x) => format!("sdl2: WindowBuildError: HeightOverflows: {}", x),
        sdl2::video::WindowBuildError::InvalidTitle(ne) => format!("sdl2: WindowBuildError: InvalidTitle: NulError: nul_position: {}", ne.nul_position()),
        sdl2::video::WindowBuildError::SdlError(sdle) => format!("sdl12: WindowBuildError: SdlError: {}", sdle),
    }
}

pub fn integer_ord_sdl_error_to_string(iose: sdl2::IntegerOrSdlError) -> String{
    match iose{
        sdl2::IntegerOrSdlError::SdlError(sdle) => format!("sdl2: IntegerOrSdlError: SdlError: {}", sdle),
        sdl2::IntegerOrSdlError::IntegerOverflows(s,u) => format!("sdl2: IntegerOrSdlError: IntegerOverflows: string = {}, value = {}", s, u),
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
