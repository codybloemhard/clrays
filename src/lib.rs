#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

extern crate sdl2;
extern crate ocl;
extern crate image;
extern crate stopwatch;
extern crate gl;

#[macro_use]
pub mod misc;
pub mod test_platform;
pub mod window;
pub mod state;
pub mod vec3;
pub mod scene;
pub mod trace_tex;
pub mod kernels;
pub mod cl_helpers;
pub mod info;
pub mod trace_processor;

pub fn test(t: test_platform::PlatformTest){
    test_platform::run_platform_test(t);
}