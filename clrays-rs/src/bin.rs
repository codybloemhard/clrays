extern crate clrays_rs;
use clrays_rs as clr;
//use clr::test_platform::PlatformTest;
use clr::window;
use clr::state;
use clr::scene::{Scene,Material,Plane,Sphere};
use clr::vec3::Vec3;
use clr::kernels::{TraceKernel};

pub fn main(){
    let mut v = vec![0usize; 10];
    for i in 0..10{
        v[i] = i;
    }

    //clr::test(PlatformTest::OpenCl1);
    let mut scene = Scene::new();
    scene.sky_col = Vec3::new(0.2, 0.2, 0.9).normalized();
    scene.sky_intensity = 0.0;
    scene.cam_pos = Vec3::zero();
    scene.cam_dir = Vec3::backward();
    scene.add_plane(Plane{
        pos: Vec3::down(),
        nor: Vec3::up(),
        mat: Material::basic(),
    });
    scene.add_sphere(Sphere{
        pos: Vec3::new(2.0, 0.0, -5.0),
        rad: 1.0,
        mat: Material::basic(),
    });

    use std::io::prelude::*;
    let mut file = std::fs::File::open("../Assets/Kernels/raytrace.cl").unwrap();
    let mut src = String::new();
    file.read_to_string(&mut src).expect("file to string aaah broken");

    let platform = ocl::Platform::default();
    let device = match ocl::Device::first(platform){
        Ok(x) => x,
        Err(_) => return,
    };
    let context = match ocl::Context::builder()
    .platform(platform)
    .devices(device.clone())
    .build(){
        Ok(x) => x,
        Err(_) => return,
    };
    let program = match ocl::Program::builder()
    .devices(device)
    .src(src)
    .build(&context){
        Ok(x) => x,
        Err(_) => return,
    };
    let queue = match ocl::Queue::new(&context, device, None){
        Ok(x) => x,
        Err(_) => return,
    };
    let mut trace_kernel = TraceKernel::new("", (960,540), &program, &queue, &mut scene);

    let mut window = window::Window::<state::StdState>::new("ClRays", 960, 540);
    window.run(window::std_input_handler);
}
