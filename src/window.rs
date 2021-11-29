use crate::state::{ State, LoopRequest, InputFn, UpdateFn };
use crate::trace_processor::TraceProcessor;
use crate::misc::Incrementable;
use crate::scene::Scene;

use stopwatch::Stopwatch;
use sdl2::event::Event;

pub struct Window{
    title: String,
    width: u32,
    height: u32,
}

impl Window
{
    pub fn new(title: &str, width: u32, height: u32) -> Self{
        Self { title: title.to_string(), width, height }
    }

    pub fn run(
            &mut self,
            input_fn: InputFn,
            update_fn: UpdateFn,
            state: &mut State,
            tracer: &mut impl TraceProcessor,
            scene: &mut Scene,
        ) -> Result<(), String>
    {
        let watch = Stopwatch::start_new();

        let contex = unpackdb!(sdl2::init(), "Could not init sdl2!");
        let video_subsystem = unpackdb!(contex.video(), "Could not get sdl video subsystem!");
        let window = unpackdb!(video_subsystem.window(&self.title, self.width, self.height)
            .position_centered()
            .opengl()
            .build(),
            "Could not create window!");
        let _gl_contex = window.gl_create_context().unwrap(); // needs to exist
        #[allow(dead_code)]
        let _gl = gl::load_with(|s| video_subsystem.gl_get_proc_address(s) as *const std::os::raw::c_void);

        let mut event_pump = unpackdb!(contex.event_pump(), "Could not get sdl event pump!");

        let mut elapsed = watch.elapsed_ms();
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
        println!("SDL+OpenGl setup time: {} ms", elapsed);
        let mut frame = 0u32;
        loop {
            // Pump all sdl2 events into vector
            let events : Vec<Event>= event_pump.poll_iter().collect();

            if input_fn(&events, scene, state) == LoopRequest::Stop { break; }
            if update_fn(0.0) == LoopRequest::Stop { break; };

            tracer.update();
            let int_tex = tracer.render(scene);
            unsafe{
                gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as i32, glw, glh, 0, gl::BGRA, gl::UNSIGNED_BYTE, int_tex.as_ptr() as *mut std::ffi::c_void);
                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
                gl::BindFramebuffer(gl::READ_FRAMEBUFFER, fbo);
                gl::BlitFramebuffer(0, 0, glw, glh, 0, glh, glw, 0, gl::COLOR_BUFFER_BIT, gl::NEAREST);
                window.gl_swap_window();
            }
            let e = watch.elapsed_ms();
            println!("Frame: {} in {} ms.", frame.inc_post(), e - elapsed);
            elapsed = e;
        }
        Ok(())
    }
}

