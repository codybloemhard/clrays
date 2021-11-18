use crate::state;
use crate::trace_processor::TraceProcessor;
use crate::misc::Incrementable;

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
        let watch = Stopwatch::start_new();
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
            .opengl()
            .build(){
            Result::Ok(x) => x,
            Result::Err(e) => return Some(window_build_error_to_string(e)),
        };
        let _gl_contex = window.gl_create_context().unwrap(); //needs to exist
        #[allow(dead_code)]
        let _gl = gl::load_with(|s| video_subsystem.gl_get_proc_address(s) as *const std::os::raw::c_void);

        let mut event_pump = match contex.event_pump(){
            Result::Ok(x) => x,
            Result::Err(e) => return Some(e),
        };
        let mut elapsed = watch.elapsed_ms();
        println!("SDL setup time: {} ms", elapsed);
        let mut texture = 0u32;
        let mut fbo = 0u32;
        let (glw,glh) = (self.width as i32, self.height as i32);
        unsafe {
            gl::GenTextures(1, &mut texture);
            gl::BindTexture(gl::TEXTURE_2D, texture);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
            gl::GenFramebuffers(1, &mut fbo);
            gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
            gl::FramebufferTexture(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, texture, 0);
        }
        let mut frame = 0;
        loop {
            handle_input(&mut event_pump, &mut self.state);
            if self.state.should_close() { break; }
            self.state.update(0.0);
            if self.state.should_close() { break; }
            self.tracer.update().expect("Couldn't update tracer!");
            let mut int_tex = self.tracer.render().unwrap().to_vec();
            unsafe{
                gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as i32, glw, glh, 0, gl::BGRA, gl::UNSIGNED_BYTE, int_tex.as_mut_ptr() as *mut std::ffi::c_void);
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::BindFramebuffer(gl::READ_FRAMEBUFFER, fbo);
                gl::BlitFramebuffer(0, 0, glw, glh, 0, glh, glw, 0, gl::COLOR_BUFFER_BIT, gl::NEAREST);
                window.gl_swap_window();
            }
            let e = watch.elapsed_ms();
            println!("Frame: {} in {} ms.", frame.inc_post(), e - elapsed);
            elapsed = e;
        }
        None
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
