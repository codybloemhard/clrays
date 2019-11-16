#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

extern crate sdl2;
extern crate ocl;

pub mod test_platform;
pub mod window;
pub mod state;
pub mod vec3;
pub mod scene;
pub mod trace_tex;

pub fn test(t: test_platform::PlatformTest){
    test_platform::run_platform_test(t);
}