#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

extern crate sdl2;

mod test_platform;
mod window;
mod state;

pub fn run(){
    let mut window = window::Window::<state::StdState>::new("ClRays", 960, 540);
    window.run(window::std_input_handler);
}

pub fn test(t: test_platform::PlatformTest){
    test_platform::run_platform_test(t);
}