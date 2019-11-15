extern crate clrays_rs;
use clrays_rs as clr;
use clr::test_platform::PlatformTest;

pub fn main(){
    clr::test(PlatformTest::OpenCl);
    //let mut window = window::Window::<state::StdState>::new("ClRays", 960, 540);
    //window.run(window::std_input_handler);
}